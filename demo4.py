import os
import gc
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix


class UserBasedCFEcommerce:
    def __init__(self, k_neighbors=50, shrinkage=20, min_common=2,
                 use_iuf=True, chunk_size=500_000, max_rows=50_000_000,
                 min_item_support=5, sim_batch_size=500, max_label_rows=10_000_000):
        """
        基于用户协同过滤的电商推荐系统(GPU加速+内存优化版)

        优化要点:
        1. 稀疏矩阵存储(CSR格式)
        2. 过滤低频商品(减少矩阵列数)
        3. GPU加速相似度计算(小批次避免显存溢出)
        4. TopK在线过滤(避免存储完整相似度矩阵)
        5. 限制标签数据读取

        Args:
            k_neighbors: 邻居数量
            shrinkage: 收缩因子
            min_common: 最小共同商品数
            use_iuf: 是否使用IUF加权
            chunk_size: CSV分块大小
            max_rows: 最大读取行数(特征期)
            min_item_support: 商品最小交互次数(过滤冷门商品)
            sim_batch_size: 相似度计算批次大小(GPU)
            max_label_rows: 标签期最大读取行数
        """
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.use_iuf = use_iuf
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.min_item_support = min_item_support
        self.sim_batch_size = sim_batch_size
        self.max_label_rows = max_label_rows

        self.start_date = datetime(2014, 11, 18)
        self.end_date = datetime(2014, 12, 17)
        self.label_date = datetime(2014, 12, 18)
        self.target_date = datetime(2014, 12, 19)

        self.behavior_weights = {
            1: 1.0,   # 浏览
            2: 3.0,   # 收藏
            3: 5.0,   # 加购
            4: 10.0   # 购买
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据结构(使用稀疏矩阵)
        self.user_id_map = {}
        self.item_id_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}
        self.rating_matrix = None  # scipy.sparse.csr_matrix
        self.user_means = None

        # TopK相似度存储(用户 -> [(邻居idx, 相似度)])
        self.user_topk_sims = {}

        print("=" * 60)
        print("用户协同过滤电商推荐系统(GPU加速+内存优化版)")
        print("=" * 60)
        print(f"配置: K邻居={k_neighbors}, 收缩={shrinkage}, 最小共同={min_common}")
        print(f"      分块={chunk_size:,}, 最大行={max_rows:,}")
        print(f"      最小商品支持={min_item_support}, 相似度批次={sim_batch_size}")
        print(f"      标签最大行={max_label_rows:,}")
        print(f"特征期: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"标签日: {self.label_date.date()}")
        print(f"预测日: {self.target_date.date()}")
        print(f"设备: {self.device}")

        # 显存监控
        if self.device.type == 'cuda':
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU显存: {total_mem:.2f} GB")

    def load_item_subset(self, item_file=r"data\tianchi_fresh_comp_train_item_online.txt"):
        """加载商品子集"""
        print("\n[1/6] 加载商品子集...")
        df = pd.read_csv(
            item_file, sep='\t', header=None,
            names=['item_id', 'item_geohash', 'item_category'],
            usecols=['item_id', 'item_category'],
            dtype='string', na_filter=False
        )
        df['item_id'] = pd.to_numeric(df['item_id'].str.strip(), errors='coerce')
        df['item_category'] = pd.to_numeric(df['item_category'].str.strip(), errors='coerce')
        df = df.dropna(subset=['item_id'])

        item_ids = set(df['item_id'].astype(np.int64))
        item_category_map = dict(zip(
            df['item_id'].astype(np.int64),
            df['item_category'].fillna(-1).astype(np.int64)
        ))

        print(f"  商品子集数量: {len(item_ids):,}")
        return item_ids, item_category_map

    def build_rating_matrix(self, item_ids, start_date, end_date, files=None):
        """构建用户-商品稀疏评分矩阵"""
        print(f"\n[2/6] 构建评分矩阵 ({start_date.date()} ~ {end_date.date()})...")

        if files is None:
            files = [
                r"data\tianchi_fresh_comp_train_user_online_partA.txt",
                r"data\tianchi_fresh_comp_train_user_online_partB.txt"
            ]

        rating_dict = defaultdict(float)
        item_support = defaultdict(int)
        user_set = set()

        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        total_rows = 0
        processed_rows = 0

        # [阶段1] 聚合评分并统计商品频率
        for file_idx, file_path in enumerate(files):
            if not os.path.exists(file_path):
                print(f"  警告: 文件不存在,跳过 {file_path}")
                continue

            print(f"\n  处理文件 [{file_idx + 1}/{len(files)}]: {file_path}")
            print(f"  最大读取行数: {self.max_rows:,}")

            chunk_iterator = pd.read_csv(
                file_path, sep='\t', header=None, names=column_names,
                usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                dtype={'user_id': 'string', 'item_id': 'string', 'behavior_type': 'int8', 'time': 'string'},
                na_filter=False, chunksize=self.chunk_size, encoding='utf-8'
            )

            pbar = tqdm(chunk_iterator, desc="    读取进度", unit="chunk")

            for chunk in pbar:
                chunk_size_actual = len(chunk)
                total_rows += chunk_size_actual

                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk = chunk.dropna(subset=['user_id', 'item_id'])
                chunk['user_id'] = chunk['user_id'].astype(np.int64)
                chunk['item_id'] = chunk['item_id'].astype(np.int64)
                chunk = chunk[chunk['item_id'].isin(item_ids)]

                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
                chunk = chunk.dropna(subset=['time'])
                chunk = chunk[(chunk['time'] >= start_date) & (chunk['time'] <= end_date)]

                if chunk.empty:
                    pbar.set_postfix({'总行': f'{total_rows:,}', '有效': processed_rows, '用户': len(user_set), '商品': len(item_support)})
                    if total_rows >= self.max_rows:
                        break
                    continue

                processed_rows += len(chunk)

                days_diff = (end_date - chunk['time']).dt.days
                time_decay = np.exp(-days_diff / 7.0)

                for row, decay in zip(chunk.itertuples(), time_decay):
                    uid = row.user_id
                    iid = row.item_id
                    btype = row.behavior_type

                    weight = self.behavior_weights.get(btype, 1.0)
                    rating_dict[(uid, iid)] += weight * decay

                    user_set.add(uid)
                    item_support[iid] += 1

                pbar.set_postfix({
                    '总行': f'{total_rows:,}',
                    '有效': processed_rows,
                    '用户': len(user_set),
                    '商品': len(item_support)
                })

                if total_rows >= self.max_rows:
                    print(f"\n  已达到最大行数限制 ({self.max_rows:,}),停止读取")
                    break

            pbar.close()

            if total_rows >= self.max_rows:
                print(f"  跳过剩余文件")
                break

        # [阶段2] 过滤低频商品
        print(f"\n  原始商品数: {len(item_support):,}")
        active_items = {iid for iid, cnt in item_support.items() if cnt >= self.min_item_support}
        print(f"  过滤后商品数: {len(active_items):,} (最小支持={self.min_item_support})")

        rating_dict = {(u, i): r for (u, i), r in rating_dict.items() if i in active_items}

        print(f"\n  加载完成: 总读取 {total_rows:,} 行, 有效 {processed_rows:,} 行")
        print(f"  用户数: {len(user_set):,}, 商品数: {len(active_items):,}")
        print(f"  评分条目: {len(rating_dict):,}")

        # [阶段3] 构建稀疏矩阵
        print("  构建稀疏矩阵...")
        users = sorted(user_set)
        items = sorted(active_items)
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.item_id_map = {m: i for i, m in enumerate(items)}
        self.inv_user_map = {i: u for u, i in self.user_id_map.items()}
        self.inv_item_map = {i: m for m, i in self.item_id_map.items()}

        n_users = len(users)
        n_items = len(items)

        rows = []
        cols = []
        data = []

        for (uid, iid), rating in tqdm(rating_dict.items(), desc="    填充矩阵"):
            if iid not in self.item_id_map:
                continue
            u_idx = self.user_id_map[uid]
            i_idx = self.item_id_map[iid]
            rows.append(u_idx)
            cols.append(i_idx)
            data.append(rating)

        coo = coo_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
        self.rating_matrix = coo.tocsr()

        # 计算用户均值
        user_sum = np.array(self.rating_matrix.sum(axis=1)).flatten()
        user_cnt = np.array((self.rating_matrix > 0).sum(axis=1)).flatten()
        global_mean = float(self.rating_matrix.data.mean()) if len(self.rating_matrix.data) > 0 else 0
        self.user_means = np.divide(user_sum, user_cnt,
                                    out=np.full(n_users, global_mean, dtype=np.float32),
                                    where=user_cnt > 0)

        sparsity = self.rating_matrix.nnz / (n_users * n_items)
        mem_mb = (self.rating_matrix.data.nbytes + self.rating_matrix.indices.nbytes +
                  self.rating_matrix.indptr.nbytes) / 1024 / 1024
        print(f"  稀疏矩阵: {self.rating_matrix.shape}, 非零元素: {self.rating_matrix.nnz:,}")
        print(f"  稀疏度: {sparsity:.6f}, 内存占用: {mem_mb:.2f} MB")

        del rating_dict, item_support
        gc.collect()

    def compute_user_similarity(self):
        """分批计算用户相似度(GPU加速,避免显存溢出)"""
        print("\n[3/6] 计算用户相似度(GPU分批加速)...")

        n_users = self.rating_matrix.shape[0]
        n_items = self.rating_matrix.shape[1]
        batch_size = self.sim_batch_size
        n_batches = (n_users + batch_size - 1) // batch_size

        print(f"  用户数: {n_users:,}, 批次数: {n_batches}, 批次大小: {batch_size}")

        # [步骤1] 预计算IUF权重并中心化
        print("  [1/4] 预计算IUF权重...")
        if self.use_iuf:
            item_pop = np.array((self.rating_matrix > 0).sum(axis=0)).flatten()
            iuf = np.log1p(n_users / (1.0 + item_pop))
            iuf_diag = csr_matrix((iuf, (range(n_items), range(n_items))), shape=(n_items, n_items))
        else:
            iuf_diag = csr_matrix(np.eye(n_items, dtype=np.float32))

        print("  [2/4] 中心化评分矩阵...")
        centered_data = []
        centered_rows = []
        centered_cols = []

        for u_idx in tqdm(range(n_users), desc="    中心化进度"):
            row = self.rating_matrix.getrow(u_idx)
            if row.nnz == 0:
                continue
            nonzero_indices = row.indices
            nonzero_values = row.data - self.user_means[u_idx]
            centered_rows.extend([u_idx] * len(nonzero_indices))
            centered_cols.extend(nonzero_indices)
            centered_data.extend(nonzero_values)

        centered = csr_matrix((centered_data, (centered_rows, centered_cols)),
                              shape=(n_users, n_items), dtype=np.float32)

        print("  [3/4] 应用IUF权重...")
        centered_iuf = centered @ iuf_diag

        print("  [4/4] 计算范数...")
        norms = np.sqrt(np.array(centered_iuf.power(2).sum(axis=1)).flatten())
        norms = np.where(norms == 0, 1.0, norms)

        # 预计算共同商品数矩阵
        print("\n  预计算共同商品数(CPU稀疏矩阵)...")
        binary_matrix = (self.rating_matrix > 0).astype(np.float32)

        # 将范数转移到GPU(只需1D数组)
        norms_gpu = torch.from_numpy(norms).to(self.device)

        # 显存监控
        if self.device.type == 'cuda':
            allocated_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
            print(f"  GPU显存使用: {allocated_gb:.2f} GB")

        # [步骤2] 分批计算相似度(每次只转移当前批次)
        print(f"\n  开始分批计算相似度(每批仅{batch_size}行)...")
        for batch_idx in tqdm(range(n_batches), desc="  批次进度"):
            start_u = batch_idx * batch_size
            end_u = min(start_u + batch_size, n_users)
            batch_len = end_u - start_u

            # 提取当前批次(稀疏->稠密->GPU)
            batch_centered_sparse = centered_iuf[start_u:end_u]
            batch_centered_dense = batch_centered_sparse.toarray()  # (batch_size, n_items)
            batch_centered_gpu = torch.from_numpy(batch_centered_dense).to(self.device)
            batch_norms_gpu = norms_gpu[start_u:end_u]

            # GPU加速计算相似度(只需将完整矩阵分块转移)
            # 分块转移完整中心化矩阵(避免一次性加载)
            all_sims = []
            transfer_batch = 2000  # 每次转移2000行
            n_transfer_batches = (n_users + transfer_batch - 1) // transfer_batch

            for tb_idx in range(n_transfer_batches):
                tb_start = tb_idx * transfer_batch
                tb_end = min(tb_start + transfer_batch, n_users)

                # 转移全量矩阵的一部分到GPU
                transfer_sparse = centered_iuf[tb_start:tb_end]
                transfer_dense = transfer_sparse.toarray()
                transfer_gpu = torch.from_numpy(transfer_dense).to(self.device)

                # 批次矩阵乘法: (batch_size, n_items) @ (transfer_size, n_items)^T
                partial_sim = torch.mm(batch_centered_gpu, transfer_gpu.T)  # (batch_size, transfer_size)
                all_sims.append(partial_sim.cpu().numpy())

                del transfer_sparse, transfer_dense, transfer_gpu
                torch.cuda.empty_cache()

            # 合并所有分块结果
            sim_batch = np.hstack(all_sims)  # (batch_size, n_users)

            # 归一化(CPU)
            batch_norms_cpu = batch_norms_gpu.cpu().numpy()
            denom = np.outer(batch_norms_cpu, norms)
            sim_batch = sim_batch / (denom + 1e-8)

            # 计算共同商品数(CPU稀疏矩阵)
            batch_binary = binary_matrix[start_u:end_u]
            common_batch = (batch_binary @ binary_matrix.T).toarray()

            # 收缩因子
            if self.shrinkage > 0:
                sim_batch = sim_batch * (common_batch / (common_batch + self.shrinkage))

            # 过滤
            sim_batch = np.where(common_batch >= self.min_common, sim_batch, 0.0)
            sim_batch = np.clip(sim_batch, 0, None)

            # 提取TopK
            for i, u_idx in enumerate(range(start_u, end_u)):
                row_sim = sim_batch[i]
                row_sim[u_idx] = 0

                k = min(self.k_neighbors, n_users - 1)
                if k > 0 and np.any(row_sim > 0):
                    topk_indices = np.argpartition(-row_sim, min(k, np.sum(row_sim > 0)))[:k]
                    topk_indices = topk_indices[row_sim[topk_indices] > 0]
                    topk_indices = topk_indices[np.argsort(-row_sim[topk_indices])]
                    self.user_topk_sims[u_idx] = [(int(idx), float(row_sim[idx]))
                                                  for idx in topk_indices]
                else:
                    self.user_topk_sims[u_idx] = []

            # 清理内存
            del sim_batch, common_batch, batch_binary, batch_centered_sparse
            del batch_centered_dense, batch_centered_gpu, batch_norms_gpu, all_sims
            gc.collect()

            # 每10批清理GPU缓存
            if (batch_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                allocated_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"    批次 {batch_idx + 1}/{n_batches}, GPU显存: {allocated_gb:.2f} GB")

        avg_neighbors = np.mean([len(v) for v in self.user_topk_sims.values()])
        print(f"\n  相似度计算完成!")
        print(f"  平均邻居数: {avg_neighbors:.1f}")

        del centered, centered_iuf, binary_matrix, iuf_diag, norms_gpu
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def predict_user_items(self, user_idx, top_n=30):
        """预测单个用户的商品评分"""
        if user_idx not in self.user_topk_sims:
            return []

        neighbors = self.user_topk_sims[user_idx]
        if not neighbors:
            return []

        n_items = self.rating_matrix.shape[1]
        pred_scores = np.full(n_items, self.user_means[user_idx], dtype=np.float32)
        already_rated = set(self.rating_matrix[user_idx].nonzero()[1])

        total_sim = 0.0
        for neighbor_idx, sim in neighbors:
            if sim <= 0:
                continue

            neighbor_row = self.rating_matrix.getrow(neighbor_idx)
            neighbor_mean = self.user_means[neighbor_idx]

            for col_idx, val in zip(neighbor_row.indices, neighbor_row.data):
                pred_scores[col_idx] += sim * (val - neighbor_mean)

            total_sim += abs(sim)

        if total_sim > 0:
            pred_scores = self.user_means[user_idx] + (pred_scores - self.user_means[user_idx]) / total_sim

        pred_scores[list(already_rated)] = -1e9

        if top_n > 0:
            valid_scores = pred_scores[pred_scores > -1e8]
            if len(valid_scores) == 0:
                return []

            topn_idx = np.argpartition(-pred_scores, min(top_n, len(pred_scores) - 1))[:top_n]
            topn_idx = topn_idx[pred_scores[topn_idx] > -1e8]
            topn_idx = topn_idx[np.argsort(-pred_scores[topn_idx])]
            return [(idx, pred_scores[idx]) for idx in topn_idx]
        else:
            return []

    def load_label_purchases(self, item_ids):
        """加载12-18的购买行为作为标签(优化版)"""
        print(f"\n[4/6] 加载标签期购买数据 ({self.label_date.date()})...")

        files = [
            r"data\tianchi_fresh_comp_train_user_online_partA.txt",
            r"data\tianchi_fresh_comp_train_user_online_partB.txt"
        ]

        purchase_pairs = set()
        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']

        total_rows = 0
        valid_purchases = 0

        known_users = set(self.user_id_map.keys())
        print(f"  训练集用户数: {len(known_users):,}")
        print(f"  最大读取行数: {self.max_label_rows:,}")

        for file_idx, file_path in enumerate(files):
            if not os.path.exists(file_path):
                print(f"  文件不存在: {file_path}")
                continue

            print(f"\n  处理文件 [{file_idx + 1}/{len(files)}]: {file_path}")

            chunk_iterator = pd.read_csv(
                file_path, sep='\t', header=None, names=column_names,
                usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                dtype={'user_id': 'string', 'item_id': 'string', 'behavior_type': 'int8', 'time': 'string'},
                na_filter=False, chunksize=self.chunk_size, encoding='utf-8'
            )

            pbar = tqdm(chunk_iterator, desc="    读取进度", unit="chunk")

            for chunk in pbar:
                chunk_size = len(chunk)
                total_rows += chunk_size

                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk = chunk.dropna(subset=['user_id', 'item_id'])
                chunk['user_id'] = chunk['user_id'].astype(np.int64)
                chunk['item_id'] = chunk['item_id'].astype(np.int64)

                chunk = chunk[chunk['user_id'].isin(known_users)]
                chunk = chunk[chunk['item_id'].isin(item_ids)]

                if chunk.empty:
                    pbar.set_postfix({'总行': f'{total_rows:,}', '有效': valid_purchases})
                    if total_rows >= self.max_label_rows:
                        break
                    continue

                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
                chunk = chunk.dropna(subset=['time'])
                chunk = chunk[chunk['time'].dt.date == self.label_date.date()]
                chunk = chunk[chunk['behavior_type'] == 4]

                if not chunk.empty:
                    for row in chunk.itertuples():
                        purchase_pairs.add((row.user_id, row.item_id))
                    valid_purchases = len(purchase_pairs)

                pbar.set_postfix({
                    '总行': f'{total_rows:,}',
                    '有效': valid_purchases
                })

                if total_rows >= self.max_label_rows:
                    print(f"\n  已达到最大行数限制 ({self.max_label_rows:,}),停止读取")
                    break

            pbar.close()

            if total_rows >= self.max_label_rows:
                print(f"  跳过剩余文件")
                break

        print(f"\n  加载完成: 总读取 {total_rows:,} 行")
        print(f"  标签期购买对数量: {len(purchase_pairs):,}")

        if len(purchase_pairs) == 0:
            print("  警告: 未找到任何购买记录!")

        return purchase_pairs

    def generate_recommendations(self, purchase_pairs, top_n=30, score_threshold=0.0):
        """生成推荐结果(在线预测)"""
        print("\n[5/6] 生成推荐(在线预测)...")

        n_users = self.rating_matrix.shape[0]
        recommendations = []
        user_rec_count = defaultdict(int)

        for u_idx in tqdm(range(n_users), desc="  生成推荐"):
            uid = self.inv_user_map[u_idx]

            pred_items = self.predict_user_items(u_idx, top_n=top_n)

            for i_idx, score in pred_items:
                if score < score_threshold:
                    continue

                iid = self.inv_item_map[i_idx]

                if (uid, iid) not in purchase_pairs:
                    recommendations.append((uid, iid))
                    user_rec_count[uid] += 1

        print(f"\n  生成推荐: {len(recommendations):,} 条")
        user_count = len(user_rec_count)
        print(f"  推荐用户数: {user_count:,}")
        if user_count > 0:
            print(f"  平均每用户推荐: {len(recommendations) / user_count:.2f} 个商品")

        return recommendations

    def save_results(self, recommendations, out_dir=r"result"):
        """保存推荐结果"""
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = os.path.join(out_dir, f"recommendation_cf_{timestamp}.txt")

        print(f"\n[6/6] 保存结果到: {out_file}")
        with open(out_file, 'w', encoding='utf-8') as f:
            for uid, iid in recommendations:
                f.write(f"{uid},{iid}\n")

        print(f"保存完成!")
        return out_file


def main():
    print("\n" + "=" * 60)
    print("开始运行用户协同过滤推荐系统")
    print("=" * 60)

    rec_sys = UserBasedCFEcommerce(
        k_neighbors=50,
        shrinkage=20,
        min_common=2,
        use_iuf=True,
        chunk_size=5_000_000,
        max_rows=50_000_000,           # 特征期最大读取5000万行
        min_item_support=5,
        sim_batch_size=1500,             # GPU批次大小(避免显存溢出)
        max_label_rows=10_000_000       # 标签期最大读取1000万行
    )

    item_ids, item_category_map = rec_sys.load_item_subset()
    rec_sys.build_rating_matrix(item_ids, rec_sys.start_date, rec_sys.end_date)
    rec_sys.compute_user_similarity()
    purchase_pairs = rec_sys.load_label_purchases(item_ids)
    recommendations = rec_sys.generate_recommendations(purchase_pairs, top_n=30, score_threshold=0.0)
    out_file = rec_sys.save_results(recommendations)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
