import os
import gc
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class UserBasedCFEcommerce:
    def __init__(self, k_neighbors=50, shrinkage=20, min_common=2,
                 use_iuf=True, chunk_size=500_000, max_rows=50_000_000):
        """
        基于用户协同过滤的电商推荐系统(GPU加速)

        Args:
            k_neighbors: 邻居数量
            shrinkage: 收缩因子
            min_common: 最小共同商品数
            use_iuf: 是否使用IUF加权
            chunk_size: 分块大小(减小到10万,更频繁打印)
            max_rows: 最大读取行数
        """
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.use_iuf = use_iuf
        self.chunk_size = chunk_size
        self.max_rows = max_rows

        self.start_date = datetime(2014, 11, 18)
        self.end_date = datetime(2014, 12, 17)
        self.label_date = datetime(2014, 12, 18)
        self.target_date = datetime(2014, 12, 19)

        self.behavior_weights = {
            1: 1.0,  # 浏览
            2: 3.0,  # 收藏
            3: 5.0,  # 加购
            4: 10.0  # 购买
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.user_id_map = {}
        self.item_id_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}
        self.rating_matrix = None
        self.sim_matrix_topk = None
        self.pred_matrix = None
        self.user_means = None

        print("=" * 60)
        print("用户协同过滤电商推荐系统(GPU加速)")
        print("=" * 60)
        print(f"配置: K邻居={k_neighbors}, 收缩={shrinkage}, 最小共同={min_common}")
        print(f"      分块大小={chunk_size:,}, 最大行数={max_rows:,}")
        print(f"特征期: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"标签日: {self.label_date.date()}")
        print(f"预测日: {self.target_date.date()}")
        print(f"设备: {self.device}")

    def load_item_subset(self, item_file=r"data\tianchi_fresh_comp_train_item_online.txt"):
        """加载商品子集"""
        print("\n[1/5] 加载商品子集...")
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
        """构建用户-商品隐式评分矩阵(带实时进度)"""
        print(f"\n[2/5] 构建评分矩阵 ({start_date.date()} ~ {end_date.date()})...")

        if files is None:
            files = [
                r"data\tianchi_fresh_comp_train_user_online_partA.txt",
                r"data\tianchi_fresh_comp_train_user_online_partB.txt"
            ]

        rating_dict = defaultdict(float)
        user_set = set()
        item_set = set()

        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        total_rows = 0
        processed_rows = 0

        for file_idx, file_path in enumerate(files):
            if not os.path.exists(file_path):
                print(f"  警告: 文件不存在,跳过 {file_path}")
                continue

            print(f"\n  处理文件 [{file_idx + 1}/{len(files)}]: {file_path}")
            print(f"  最大读取行数: {self.max_rows:,} (达到后自动停止)")

            chunk_iterator = pd.read_csv(
                file_path, sep='\t', header=None, names=column_names,
                usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                dtype={'user_id': 'string', 'item_id': 'string', 'behavior_type': 'int8', 'time': 'string'},
                na_filter=False, chunksize=self.chunk_size, encoding='utf-8'
            )

            # 使用tqdm,不设置总长度(动态更新)
            pbar = tqdm(chunk_iterator, desc="    读取进度", unit="chunk")

            for chunk in pbar:
                chunk_size_actual = len(chunk)
                total_rows += chunk_size_actual

                # 解析并过滤
                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk = chunk.dropna(subset=['user_id', 'item_id'])
                chunk['user_id'] = chunk['user_id'].astype(np.int64)
                chunk['item_id'] = chunk['item_id'].astype(np.int64)

                # 只保留商品子集
                chunk = chunk[chunk['item_id'].isin(item_ids)]

                # 时间过滤
                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
                chunk = chunk.dropna(subset=['time'])
                chunk = chunk[(chunk['time'] >= start_date) & (chunk['time'] <= end_date)]

                if chunk.empty:
                    continue

                processed_rows += len(chunk)

                # 计算时间衰减
                days_diff = (end_date - chunk['time']).dt.days
                time_decay = np.exp(-days_diff / 7.0)

                # 聚合评分
                for row, decay in zip(chunk.itertuples(), time_decay):
                    uid = row.user_id
                    iid = row.item_id
                    behavior = row.behavior_type
                    weight = self.behavior_weights.get(behavior, 0.0)
                    rating_dict[(uid, iid)] += weight * decay
                    user_set.add(uid)
                    item_set.add(iid)

                # 更新进度条后缀信息
                pbar.set_postfix({
                    '总行': f'{total_rows:,}',
                    '有效': f'{processed_rows:,}',
                    '用户': f'{len(user_set):,}',
                    '商品': f'{len(item_set):,}'
                })

                # 检查是否达到最大行数
                if total_rows >= self.max_rows:
                    print(f"\n  已达到最大行数限制 ({self.max_rows:,}),停止读取")
                    pbar.close()
                    break

            pbar.close()

            if total_rows >= self.max_rows:
                print(f"  跳过剩余文件")
                break

        print(f"\n  加载完成: 总读取 {total_rows:,} 行, 有效 {processed_rows:,} 行")
        print(f"  用户数: {len(user_set):,}, 商品数: {len(item_set):,}")
        print(f"  评分条目: {len(rating_dict):,}")

        # 构建映射
        print("  构建ID映射...")
        users = sorted(user_set)
        items = sorted(item_set)
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.item_id_map = {m: i for i, m in enumerate(items)}
        self.inv_user_map = {i: u for u, i in self.user_id_map.items()}
        self.inv_item_map = {i: m for m, i in self.item_id_map.items()}

        # 构建矩阵
        print("  构建评分矩阵...")
        n_users = len(users)
        n_items = len(items)
        self.rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)

        for (uid, iid), rating in tqdm(rating_dict.items(), desc="    填充矩阵"):
            u_idx = self.user_id_map[uid]
            i_idx = self.item_id_map[iid]
            self.rating_matrix[u_idx, i_idx] = rating

        # 用户均值
        user_sum = self.rating_matrix.sum(axis=1)
        user_cnt = (self.rating_matrix > 0).sum(axis=1)
        global_mean = float(self.rating_matrix[self.rating_matrix > 0].mean()) if processed_rows > 0 else 0
        self.user_means = np.divide(user_sum, user_cnt,
                                    out=np.full(n_users, global_mean, dtype=np.float32),
                                    where=user_cnt > 0)

        sparsity = (self.rating_matrix > 0).sum() / (n_users * n_items)
        print(f"  评分矩阵: {self.rating_matrix.shape}, 稀疏度: {sparsity:.4f}")

    def compute_user_similarity(self):
        """计算用户相似度(GPU加速)"""
        print("\n[3/5] 计算用户相似度(GPU)...")

        print("  [1/6] 转移数据到GPU...")
        R = torch.from_numpy(self.rating_matrix).to(self.device)
        mask = (R > 0).float()
        user_means = torch.from_numpy(self.user_means).to(self.device).unsqueeze(1)

        print("  [2/6] 均值中心化...")
        centered = (R - user_means) * mask

        # IUF加权
        if self.use_iuf:
            print("  [3/6] IUF加权...")
            item_pop = mask.sum(0)
            U = mask.size(0)
            iuf = torch.log1p(U / (1.0 + item_pop))
            centered = centered * iuf
        else:
            print("  [3/6] 跳过IUF加权")

        # 余弦相似度
        print("  [4/6] 计算余弦相似度...")
        norms = torch.norm(centered, dim=1)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        sim = centered @ centered.t()
        denom = torch.outer(norms, norms)
        sim = sim / (denom + 1e-8)

        # 共同商品数
        print("  [5/6] 计算共同商品数...")
        common = (mask @ mask.t())

        # 收缩因子
        if self.shrinkage > 0:
            print(f"  [6/6] 应用收缩因子(shrinkage={self.shrinkage})...")
            sim = sim * (common / (common + self.shrinkage))

        # 过滤
        sim = torch.where(common >= self.min_common, sim, torch.zeros_like(sim))
        sim = torch.clamp(sim, min=0)
        sim.fill_diagonal_(0.0)

        # TopK稀疏化
        print(f"  [7/7] TopK稀疏化(K={self.k_neighbors})...")
        k = min(self.k_neighbors, sim.size(1) - 1)
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
        pruned = torch.zeros_like(sim)
        pruned.scatter_(1, topk_idx, topk_vals)

        self.sim_matrix_topk = pruned

        avg_neighbors = (pruned > 0).sum(dim=1).float().mean().item()
        print(f"\n  相似度矩阵: {pruned.shape}")
        print(f"  平均邻居数: {avg_neighbors:.1f}")

    def predict_all(self):
        """批量预测购买概率"""
        print("\n[4/5] 批量预测购买概率(GPU)...")

        R = torch.from_numpy(self.rating_matrix).to(self.device)
        mask = (R > 0).float()
        user_means = torch.from_numpy(self.user_means).to(self.device).unsqueeze(1)

        print("  [1/3] 中心化...")
        centered = (R - user_means) * mask

        print("  [2/3] 相似度加权求和...")
        S = self.sim_matrix_topk
        numer = S @ centered
        denom = torch.clamp(S.sum(dim=1, keepdim=True), min=1e-6)

        print("  [3/3] 计算预测评分...")
        pred = user_means + numer / denom
        max_rating = float(R.max().item())
        pred = torch.clamp(pred, 0.0, max_rating)

        self.pred_matrix = pred

        print(f"\n  预测矩阵: {pred.shape}")

    def load_label_purchases(self, item_ids):
        """加载12-18的购买行为作为标签"""
        print(f"\n[3.5/5] 加载标签期购买数据 ({self.label_date.date()})...")

        files = [
            r"data\tianchi_fresh_comp_train_user_online_partA.txt",
            r"data\tianchi_fresh_comp_train_user_online_partB.txt"
        ]

        purchase_pairs = set()
        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']

        for file_path in files:
            if not os.path.exists(file_path):
                continue

            for chunk in pd.read_csv(
                    file_path, sep='\t', header=None, names=column_names,
                    usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                    chunksize=self.chunk_size, dtype='string', na_filter=False
            ):
                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk['behavior_type'] = pd.to_numeric(chunk['behavior_type'].str.strip(), errors='coerce')
                chunk['time'] = pd.to_datetime(chunk['time'].str.strip(), format='%Y-%m-%d %H', errors='coerce')

                chunk = chunk.dropna()
                chunk = chunk[chunk['behavior_type'] == 4]
                chunk = chunk[chunk['item_id'].isin(item_ids)]
                chunk = chunk[chunk['time'].dt.date == self.label_date.date()]

                for row in chunk.itertuples():
                    purchase_pairs.add((int(row.user_id), int(row.item_id)))

        print(f"  标签期购买对数量: {len(purchase_pairs):,}")
        return purchase_pairs

    def generate_recommendations(self, purchase_pairs, top_n=30, threshold_percentile=40):
        """生成推荐结果"""
        print("\n[5/5] 生成推荐...")

        pred = self.pred_matrix.cpu().numpy()
        R_train = self.rating_matrix

        n_users, n_items = pred.shape

        # 计算阈值
        all_scores = pred[R_train == 0]
        if len(all_scores) > 0:
            score_threshold = float(np.percentile(all_scores, threshold_percentile))
        else:
            score_threshold = 0.0

        print(f"  评分阈值(百分位{threshold_percentile}): {score_threshold:.4f}")

        recommendations = []
        user_rec_count = defaultdict(int)

        for u_idx in tqdm(range(n_users), desc="  生成推荐"):
            uid = self.inv_user_map[u_idx]
            already = R_train[u_idx] > 0
            user_scores = pred[u_idx].copy()
            user_scores[already] = -1e9

            # 候选商品
            candidates = np.where(user_scores >= score_threshold)[0]
            if len(candidates) == 0:
                continue

            # 排序取TopN
            top_idx = candidates[np.argsort(-user_scores[candidates])][:top_n]

            for i_idx in top_idx:
                iid = self.inv_item_map[i_idx]
                if (uid, iid) not in purchase_pairs:  # 避免推荐已购商品
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
                f.write(f"{uid}\t{iid}\n")

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
        chunk_size=500_000,  # 减小分块,更频繁打印
        max_rows=50_000_000
    )

    item_ids, item_category_map = rec_sys.load_item_subset()
    rec_sys.build_rating_matrix(item_ids, rec_sys.start_date, rec_sys.end_date)
    rec_sys.compute_user_similarity()
    purchase_pairs = rec_sys.load_label_purchases(item_ids)
    rec_sys.predict_all()
    recommendations = rec_sys.generate_recommendations(purchase_pairs, top_n=30, threshold_percentile=40)
    out_file = rec_sys.save_results(recommendations)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
