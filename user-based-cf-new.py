import os
import urllib.request
import zipfile
import time
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict


class UserBasedCFOptimized:
    def __init__(self,
                 k_neighbors=80,
                 shrinkage=30,
                 min_common=3,
                 use_iuf=True,
                 keep_neg_sims=True,
                 topn=10,
                 sim_batch=None):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.use_iuf = use_iuf
        self.keep_neg_sims = keep_neg_sims
        self.topn = topn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")
        # 数据
        self.ratings_df = None
        self.movies_df = None
        # 映射
        self.user_id_map = {}
        self.movie_id_map = {}
        self.inv_user_map = {}
        self.inv_movie_map = {}
        # 矩阵
        self.train_mat = None  # numpy
        self.test_df = None
        # 预测缓存
        self.pred_matrix = None
        self.user_means = None
        self.sim_matrix_topk = None  # 稀疏后的相似度(稠密存储但大量0)
        self.global_mean = None

    def download_and_load_data(self, data_dir='./data'):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, 'ml-100k.zip')
        extract_path = os.path.join(data_dir, 'ml-100k')
        data_file = os.path.join(extract_path, 'u.data')
        if not os.path.exists(data_file):
            print("下载数据集...")
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, zip_path)
            print("解压数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
        columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(data_file, sep='\t', names=columns, encoding='latin-1')
        movie_file = os.path.join(extract_path, 'u.item')
        movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date',
                         'imdb_url'] + [f'genre_{i}' for i in range(19)]
        self.movies_df = pd.read_csv(movie_file, sep='|', names=movie_columns,
                                     encoding='latin-1', usecols=['movie_id', 'title'])
        print(f"加载评分: {len(self.ratings_df)}")

    def prepare_data(self, test_size=0.2, seed=42):
        users = sorted(self.ratings_df.user_id.unique())
        movies = sorted(self.ratings_df.movie_id.unique())
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.movie_id_map = {m: i for i, m in enumerate(movies)}
        self.inv_user_map = {i: u for u, i in self.user_id_map.items()}
        self.inv_movie_map = {i: m for m, i in self.movie_id_map.items()}

        self.ratings_df['u_idx'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['m_idx'] = self.ratings_df['movie_id'].map(self.movie_id_map)

        train_df, test_df = train_test_split(self.ratings_df, test_size=test_size, random_state=seed)
        self.test_df = test_df.reset_index(drop=True)

        n_u = len(users)
        n_m = len(movies)
        self.train_mat = np.zeros((n_u, n_m), dtype=np.float32)
        for r in train_df.itertuples():
            self.train_mat[r.u_idx, r.m_idx] = r.rating
        self.global_mean = float(self.train_mat[self.train_mat > 0].mean())
        user_sum = self.train_mat.sum(axis=1)
        user_cnt = (self.train_mat > 0).sum(axis=1)
        self.user_means = np.divide(user_sum, user_cnt, out=np.full(n_u, self.global_mean, dtype=np.float32),
                                    where=user_cnt > 0)
        print(f"用户数:{n_u}  物品数:{n_m}")

    def compute_similarity(self):
        print("计算用户相似度(GPU)...")
        start = time.time()
        R = torch.from_numpy(self.train_mat).to(self.device)  # U x I
        mask = (R > 0).float()
        user_means = torch.from_numpy(self.user_means).to(self.device).unsqueeze(1)
        centered = (R - user_means) * mask  # 均值中心化

        # IUF
        if self.use_iuf:
            item_pop = mask.sum(0)  # 每个物品被多少用户评分
            # IUF = log(1 + U / (1 + n_i))
            U = mask.size(0)
            iuf = torch.log1p(U / (1.0 + item_pop))
            centered = centered * iuf  # 列缩放

        # 计算向量范数
        norms = torch.norm(centered, dim=1)  # U
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        # 余弦相似度: (C @ C^T) / (||c_u|| * ||c_v||)
        sim = centered @ centered.t()
        denom = torch.outer(norms, norms)
        sim = sim / denom

        # 共同评分数
        common = (mask @ mask.t())  # U x U
        # 显著性 & shrinkage
        if self.shrinkage > 0:
            sim = sim * (common / (common + self.shrinkage))
        # 去除共同评分过少
        sim = torch.where(common >= self.min_common, sim, torch.zeros_like(sim))
        # 是否保留负相似度
        if not self.keep_neg_sims:
            sim = torch.clamp(sim, min=0)
        sim.fill_diagonal_(0.0)

        # Top-K 稀疏化
        k = min(self.k_neighbors, sim.size(1) - 1)
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
        pruned = torch.zeros_like(sim)
        pruned.scatter_(1, topk_idx, topk_vals)
        self.sim_matrix_topk = pruned  # 仍在 GPU
        end = time.time()
        print(f"相似度完成，用时 {end - start:.2f}s")

    def predict_all(self):
        print("批量预测矩阵(GPU)...")
        start = time.time()
        if self.sim_matrix_topk is None:
            raise RuntimeError("需先计算相似度")
        R = torch.from_numpy(self.train_mat).to(self.device)
        mask = (R > 0).float()
        user_means = torch.from_numpy(self.user_means).to(self.device).unsqueeze(1)
        centered = (R - user_means) * mask  # U x I

        S = self.sim_matrix_topk  # U x U
        # 分子: S * centered
        numer = S @ centered  # U x I
        denom = torch.clamp(S.abs().sum(dim=1, keepdim=True), min=1e-6)  # U x 1
        pred = user_means + numer / denom
        # 限制范围
        pred = torch.clamp(pred, 1.0, 5.0)
        # 已评分位置用原评分覆盖（可选：也可不覆盖用于评分预测评估）
        filled = pred.clone()
        filled[mask > 0] = R[mask > 0]
        self.pred_matrix = filled.detach().cpu().numpy()
        print(f"预测完成，用时 {time.time() - start:.2f}s")

    def evaluate_rating(self):
        # 评分预测精度(RMSE/MAE) 使用 test 集中条目预测
        print("评估评分预测...")
        if self.pred_matrix is None:
            raise RuntimeError("先调用 predict_all()")
        y_true = []
        y_pred = []
        for row in self.test_df.itertuples():
            u = row.u_idx
            i = row.m_idx
            # 如果训练集中已有评分，仍然用模型预测(不使用覆盖版本)更公平:
            y_true.append(row.rating)
            y_pred.append(self.pred_matrix[u, i])
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    def evaluate_topn(self, threshold=4.0):
        print("评估Top-N推荐...")
        if self.pred_matrix is None:
            raise RuntimeError("先调用 predict_all()")
        R_train = self.train_mat
        pred = self.pred_matrix
        n_users, n_items = pred.shape

        # 真实喜欢集合(基于测试集)
        test_group = self.test_df.groupby('u_idx')
        precision_list = []
        recall_list = []
        coverage_items = set()

        # 物品流行度(训练)
        item_pop = (R_train > 0).sum(axis=0)
        pop_list = []

        for u, grp in test_group:
            # 用户在测试集中喜欢的物品
            liked_true = set(grp[grp.rating >= threshold].m_idx.values)
            if not liked_true:
                continue
            # 候选：未在训练集中出现的物品
            already = R_train[u] > 0
            user_scores = pred[u].copy()
            user_scores[already] = -1e9  # 屏蔽已看
            top_idx = np.argpartition(-user_scores, self.topn)[:self.topn]
            top_idx = top_idx[np.argsort(-user_scores[top_idx])]
            rec_set = set(top_idx.tolist())
            hit = liked_true & rec_set
            precision_list.append(len(hit) / self.topn)
            recall_list.append(len(hit) / len(liked_true))
            coverage_items.update(rec_set)
            pop_list.append(item_pop[list(rec_set)].mean() if rec_set else 0)

        precision = float(np.mean(precision_list)) if precision_list else 0.0
        recall = float(np.mean(recall_list)) if recall_list else 0.0
        coverage = len(coverage_items) / n_items
        popularity = float(np.mean(pop_list)) if pop_list else 0.0
        return precision, recall, coverage, popularity

    def run(self):
        self.download_and_load_data()
        self.prepare_data()
        self.compute_similarity()
        self.predict_all()
        rmse, mae = self.evaluate_rating()
        precision, recall, coverage, popularity = self.evaluate_topn()
        print("\n=== 指标 ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Precision@{self.topn}: {precision:.4f}")
        print(f"Recall@{self.topn}: {recall:.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Popularity: {popularity:.2f}")


if __name__ == "__main__":
    model = UserBasedCFOptimized(
        k_neighbors=14,
        shrinkage=30,
        min_common=3,
        use_iuf=True,
        keep_neg_sims=True,
        topn=10
    )
    model.run()
