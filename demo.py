import os
import gc
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class HybridRecommendationSystem:
    def __init__(self, chunk_size=1_000_000, max_rows=50_000_000):
        """
        混合推荐系统(特征工程 + LightGBM + 信号分层)

        Args:
            chunk_size: 分块读取大小
            max_rows: 最大读取行数
        """
        self.chunk_size = chunk_size
        self.max_rows = max_rows

        # 训练日期范围: 2014-11-18 ~ 2014-12-17 (不包含12-18)
        self.start_date = datetime(2014, 11, 18)
        self.end_date = datetime(2014, 12, 17)
        self.label_date = datetime(2014, 12, 18)  # 用12-18的购买作为标签
        self.target_date = datetime(2014, 12, 19)

        # 行为权重配置(强信号优先)
        self.behavior_weights = {
            1: 1.0,  # 浏览(基础信号)
            2: 3.0,  # 收藏(中等信号)
            3: 5.0,  # 加购(强信号)
            4: 10.0  # 购买(最强信号)
        }

        print("=" * 60)
        print("混合推荐系统 - LightGBM + 特征工程 + 信号分层")
        print("=" * 60)
        print(f"配置: 分块大小={chunk_size:,}, 最大行数={max_rows:,}")
        print(f"特征期: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"标签日: {self.label_date.date()}")
        print(f"预测日: {self.target_date.date()}")

    def load_item_subset(self, item_file=r"data\tianchi_fresh_comp_train_item_online.txt"):
        """加载商品子集P"""
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

    def load_and_aggregate_behaviors(self, item_ids, item_category_map, files=None,
                                     start_date=None, end_date=None, label_mode=False):
        """
        加载行为数据并聚合特征

        Args:
            label_mode: True=加载标签期数据, False=加载特征期数据
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        mode_name = "标签期" if label_mode else "特征期"
        print(f"\n[2/6] 加载{mode_name}用户行为 ({start_date.date()} ~ {end_date.date()})...")

        if files is None:
            files = [
                r"data\tianchi_fresh_comp_train_user_online_partA.txt",
                r"data\tianchi_fresh_comp_train_user_online_partB.txt"
            ]

        # 特征聚合字典
        features_dict = defaultdict(lambda: {
            'clicks': 0, 'collects': 0, 'carts': 0, 'purchases': 0,
            'last_click_time': None, 'last_collect_time': None, 'last_cart_time': None,
            'active_days': set(),
            'weighted_score': 0.0
        })

        # 用户/商品/类目统计
        user_stats = defaultdict(int)
        item_stats = defaultdict(int)
        category_stats = defaultdict(lambda: {'total': 0, 'buy': 0})

        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        total_rows = 0
        processed_rows = 0

        for file_idx, file_path in enumerate(files):
            print(f"\n  处理文件 [{file_idx + 1}/{len(files)}]: {file_path}")

            for chunk in pd.read_csv(
                    file_path, sep='\t', header=None, names=column_names,
                    usecols=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'],
                    chunksize=self.chunk_size, dtype='string', na_filter=False
            ):
                chunk_size = len(chunk)
                total_rows += chunk_size

                # 类型转换
                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk['behavior_type'] = pd.to_numeric(chunk['behavior_type'].str.strip(), errors='coerce')
                chunk['item_category'] = pd.to_numeric(chunk['item_category'].str.strip(), errors='coerce')
                chunk['time'] = pd.to_datetime(chunk['time'].str.strip(), format='%Y-%m-%d %H', errors='coerce')

                chunk = chunk.dropna().copy()
                chunk = chunk[(chunk['behavior_type'] >= 1) & (chunk['behavior_type'] <= 4)]
                chunk = chunk[chunk['item_id'].isin(item_ids)]

                # 时间范围过滤
                chunk = chunk[(chunk['time'] >= start_date) & (chunk['time'] <= end_date)]

                valid_size = len(chunk)
                processed_rows += valid_size

                if valid_size == 0:
                    continue

                chunk['user_id'] = chunk['user_id'].astype(np.int64)
                chunk['item_id'] = chunk['item_id'].astype(np.int64)
                chunk['behavior_type'] = chunk['behavior_type'].astype(np.int8)
                chunk['item_category'] = chunk['item_category'].astype(np.int64)

                # 聚合特征
                for _, row in chunk.iterrows():
                    uid, iid, btype, cat, time = row['user_id'], row['item_id'], row['behavior_type'], row[
                        'item_category'], row['time']
                    key = (uid, iid)

                    # 行为频次统计
                    if btype == 1:
                        features_dict[key]['clicks'] += 1
                        if features_dict[key]['last_click_time'] is None or time > features_dict[key][
                            'last_click_time']:
                            features_dict[key]['last_click_time'] = time
                    elif btype == 2:
                        features_dict[key]['collects'] += 1
                        if features_dict[key]['last_collect_time'] is None or time > features_dict[key][
                            'last_collect_time']:
                            features_dict[key]['last_collect_time'] = time
                    elif btype == 3:
                        features_dict[key]['carts'] += 1
                        if features_dict[key]['last_cart_time'] is None or time > features_dict[key]['last_cart_time']:
                            features_dict[key]['last_cart_time'] = time
                    elif btype == 4:
                        features_dict[key]['purchases'] += 1

                    # 活跃天数
                    features_dict[key]['active_days'].add(time.date())

                    # 时间衰减权重得分
                    days_ago = (end_date - time).days
                    decay_weight = np.exp(-days_ago / 5.0)  # 5天半衰期
                    features_dict[key]['weighted_score'] += self.behavior_weights[btype] * decay_weight

                    # 用户/商品/类目统计
                    user_stats[uid] += 1
                    item_stats[iid] += 1
                    category_stats[cat]['total'] += 1
                    if btype == 4:
                        category_stats[cat]['buy'] += 1

                print(f"    已处理 {processed_rows:,}/{total_rows:,} 有效行", end='\r')

                if total_rows >= self.max_rows:
                    print(f"\n  已达到最大行数限制")
                    break

            if total_rows >= self.max_rows:
                break

        print(f"\n\n  加载完成: 总读取 {total_rows:,} 行, 有效 {processed_rows:,} 行")
        print(f"  聚合用户-商品对: {len(features_dict):,}")

        # 如果是标签模式,只返回购买对
        if label_mode:
            purchase_pairs = {key for key, feats in features_dict.items() if feats['purchases'] > 0}
            print(f"  购买对数量: {len(purchase_pairs):,}")
            return purchase_pairs

        # 构建特征DataFrame
        print("\n  构建特征矩阵...")
        records = []
        for (uid, iid), feats in tqdm(features_dict.items(), desc="  构建特征"):
            # 基础特征
            rec = {
                'user_id': uid,
                'item_id': iid,
                'clicks': feats['clicks'],
                'collects': feats['collects'],
                'carts': feats['carts'],
            }

            # 时间衰减特征
            rec['last_click_days'] = (end_date - feats['last_click_time']).days if feats['last_click_time'] else 999
            rec['last_collect_days'] = (end_date - feats['last_collect_time']).days if feats[
                'last_collect_time'] else 999
            rec['last_cart_days'] = (end_date - feats['last_cart_time']).days if feats['last_cart_time'] else 999

            # 活跃度特征
            rec['active_days_count'] = len(feats['active_days'])
            rec['is_active_last_1day'] = 1 if any(d == end_date.date() for d in feats['active_days']) else 0
            rec['is_active_last_3days'] = 1 if any(
                (end_date - timedelta(days=2)).date() <= d <= end_date.date() for d in feats['active_days']) else 0

            # 加权得分
            rec['weighted_score'] = feats['weighted_score']

            # 用户/商品热度
            rec['user_total_behaviors'] = user_stats[uid]
            rec['item_popularity'] = item_stats[iid]

            # 类目购买率
            cat = item_category_map.get(iid, -1)
            if cat in category_stats and category_stats[cat]['total'] > 0:
                rec['category_buy_rate'] = category_stats[cat]['buy'] / category_stats[cat]['total']
            else:
                rec['category_buy_rate'] = 0.0

            # 信号强度分类(基于加购和收藏)
            if feats['carts'] > 0:
                rec['signal_level'] = 2  # 强信号
            elif feats['collects'] > 0:
                rec['signal_level'] = 1  # 中等信号
            else:
                rec['signal_level'] = 0  # 弱信号

            records.append(rec)

        df = pd.DataFrame(records)
        print(f"  特征矩阵: {df.shape}")
        return df

    def create_labels(self, df, purchase_pairs):
        """
        构造标签: 使用12-18的购买行为作为标签

        Args:
            df: 特征DataFrame
            purchase_pairs: 12-18当天的购买对集合
        """
        print("\n[3/6] 构造训练标签...")

        # 标记标签
        df['label'] = df.apply(lambda x: 1 if (x['user_id'], x['item_id']) in purchase_pairs else 0, axis=1)

        pos_count = df['label'].sum()
        neg_count = len(df) - pos_count

        print(f"  正样本: {pos_count:,} ({pos_count / len(df) * 100:.2f}%)")
        print(f"  负样本: {neg_count:,} ({neg_count / len(df) * 100:.2f}%)")

        # 样本不平衡处理: 下采样负样本(保留正样本的5倍)
        if neg_count > pos_count * 5:
            print(f"  执行负样本下采样(保留正样本的5倍)...")
            pos_df = df[df['label'] == 1]
            neg_df = df[df['label'] == 0].sample(n=min(pos_count * 5, neg_count), random_state=42)
            df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"  下采样后样本数: {len(df):,}")

        return df

    def train_model(self, df):
        """
        训练LightGBM模型

        特征: 14维(行为频次+时间衰减+活跃度+热度+类目+加权得分+信号强度)
        """
        print("\n[4/6] 训练LightGBM模型...")

        # 特征列(移除purchases避免标签泄露)
        feature_cols = [
            'clicks', 'collects', 'carts',
            'last_click_days', 'last_collect_days', 'last_cart_days',
            'active_days_count', 'is_active_last_1day', 'is_active_last_3days',
            'weighted_score', 'user_total_behaviors', 'item_popularity',
            'category_buy_rate', 'signal_level'
        ]

        X = df[feature_cols]
        y = df['label']

        # 训练集/验证集划分
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"  训练集: {len(X_train):,}, 验证集: {len(X_val):,}")

        # LightGBM数据集
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # 检测GPU支持
        try:
            gpu_supported = 'gpu' in lgb.build_info().get('bin_constr', '')
        except:
            gpu_supported = False

        # 模型参数
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'seed': 42,
            'verbose': -1
        }

        if gpu_supported:
            print("  检测到GPU支持,启用GPU加速")
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })

        # 训练模型
        print("  开始训练...")
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50)
            ]
        )

        # 输出最佳性能
        best_iter = model.best_iteration
        print(f"\n  训练完成!")
        print(f"  最佳迭代: {best_iter}")

        # 特征重要性
        print("\n  Top 10 重要特征:")
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']:25s}: {row['importance']:.1f}")

        return model, feature_cols

    def generate_predictions(self, model, feature_cols, df_all, item_category_map, top_n=30):
        """
        生成预测结果(动态阈值策略)

        策略:
        1. 强信号(signal_level=2): 概率>0.1即推荐
        2. 中等信号(signal_level=1): 概率>0.2推荐
        3. 弱信号(signal_level=0): 概率>0.3推荐
        """
        print("\n[5/6] 生成预测...")

        # 预测概率
        X = df_all[feature_cols]
        y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
        df_all['pred_proba'] = y_pred_proba

        # 动态阈值过滤
        print("  应用动态阈值策略...")
        conditions = [
            (df_all['signal_level'] == 2) & (df_all['pred_proba'] > 0.1),  # 强信号
            (df_all['signal_level'] == 1) & (df_all['pred_proba'] > 0.2),  # 中等信号
            (df_all['signal_level'] == 0) & (df_all['pred_proba'] > 0.3),  # 弱信号
        ]
        df_filtered = df_all[np.logical_or.reduce(conditions)].copy()

        print(f"  过滤后候选数: {len(df_filtered):,}")

        # 如果过滤后为空,降低阈值
        if len(df_filtered) == 0:
            print("  警告: 过滤后无候选,降低阈值至0.05...")
            df_filtered = df_all[df_all['pred_proba'] > 0.05].copy()
            print(f"  新候选数: {len(df_filtered):,}")

        # 按用户分组取Top-N
        print("  按用户分组并排序...")
        recommendations = []

        for uid, group in tqdm(df_filtered.groupby('user_id'), desc="  生成推荐"):
            # 按概率降序排序
            group_sorted = group.sort_values('pred_proba', ascending=False)

            # 动态Top-N: 根据用户活跃度调整
            user_activity = group_sorted['user_total_behaviors'].iloc[0]
            dynamic_n = min(top_n, max(5, int(user_activity / 20)))

            for _, row in group_sorted.head(dynamic_n).iterrows():
                recommendations.append((int(row['user_id']), int(row['item_id'])))

        print(f"\n  生成推荐: {len(recommendations):,} 条")
        user_count = len(set(uid for uid, _ in recommendations))
        print(f"  推荐用户数: {user_count:,}")
        if user_count > 0:
            print(f"  平均每用户推荐: {len(recommendations) / user_count:.2f} 个商品")

        return recommendations

    def save_results(self, recommendations, out_dir=r"result"):
        """保存推荐结果"""
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = os.path.join(out_dir, f"recommendation_{timestamp}.txt")

        print(f"\n[6/6] 保存结果到: {out_file}")
        with open(out_file, 'w', encoding='utf-8') as f:
            for uid, iid in recommendations:
                f.write(f"{uid}\t{iid}\n")

        print(f"保存完成!")
        return out_file


def main():
    print("\n" + "=" * 60)
    print("开始运行混合推荐系统")
    print("=" * 60)

    # 初始化系统
    rec_sys = HybridRecommendationSystem(
        chunk_size=1_000_000,
        max_rows=50_000_000
    )

    # [1] 加载商品子集
    item_ids, item_category_map = rec_sys.load_item_subset()

    # [2] 加载特征期行为数据(11-18 ~ 12-17)
    df_features = rec_sys.load_and_aggregate_behaviors(
        item_ids, item_category_map,
        start_date=rec_sys.start_date,
        end_date=rec_sys.end_date,
        label_mode=False
    )

    # [2.5] 加载标签期购买数据(12-18)
    purchase_pairs = rec_sys.load_and_aggregate_behaviors(
        item_ids, item_category_map,
        start_date=rec_sys.label_date,
        end_date=rec_sys.label_date,
        label_mode=True
    )

    # [3] 构造标签
    df_labeled = rec_sys.create_labels(df_features.copy(), purchase_pairs)

    # [4] 训练模型
    model, feature_cols = rec_sys.train_model(df_labeled)

    # [5] 生成预测(使用全量特征数据)
    recommendations = rec_sys.generate_predictions(
        model, feature_cols, df_features, item_category_map, top_n=30
    )

    # [6] 保存结果
    out_file = rec_sys.save_results(recommendations)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
