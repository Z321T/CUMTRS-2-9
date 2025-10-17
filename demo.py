import os
import gc
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class TimeWindowRecommendationSystem:
    def __init__(self, max_span=7, chunk_size=500_000, max_rows=50_000_000):
        """
        基于时间窗口的监督学习推荐系统(增强版)

        Args:
            max_span: 最大时间窗口(天),默认7天
            chunk_size: 分块读取大小
            max_rows: 最大读取行数
        """
        self.max_span = max_span
        self.chunk_size = chunk_size
        self.max_rows = max_rows

        # 训练日期范围: 2014-11-18 ~ 2014-12-18
        self.start_date = datetime(2014, 11, 18)
        self.end_date = datetime(2014, 12, 18)
        self.target_date = datetime(2014, 12, 19)

        print("=" * 50)
        print("时间窗口监督学习推荐系统(增强版)")
        print("=" * 50)
        print(f"配置: 最大时间窗口={max_span}天, 分块大小={chunk_size:,}")
        print(f"训练期: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"预测日: {self.target_date.date()}")

    def load_item_subset(self, item_file=r"data\tianchi_fresh_comp_train_item_online.txt"):
        """加载商品子集"""
        print("\n加载商品子集...")
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

        print(f"商品子集数量: {len(item_ids):,}")
        return item_ids, item_category_map

    def load_behavior_data(self, item_ids, files=None):
        """加载并按日期分组行为数据,增加用户/商品统计"""
        print("\n加载用户行为数据...")
        if files is None:
            files = [
                r"data\tianchi_fresh_comp_train_user_online_partA.txt",
                r"data\tianchi_fresh_comp_train_user_online_partB.txt"
            ]

        # 按日期分组的行为数据: {date: {(user_id, item_id): [浏览,收藏,加购,购买]}}
        daily_behaviors = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

        # 用户活跃度统计
        user_activity = defaultdict(lambda: {'total': 0, 'buy': 0, 'cart': 0, 'fav': 0})
        # 商品热度统计
        item_popularity = defaultdict(lambda: {'total': 0, 'buy': 0, 'cart': 0, 'fav': 0})
        # 类目统计
        category_stats = defaultdict(lambda: {'total': 0, 'buy': 0})

        column_names = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        total_rows = 0
        processed_rows = 0

        for file_idx, file_path in enumerate(files):
            print(f"\n处理文件 [{file_idx + 1}/{len(files)}]: {file_path}")
            chunk_count = 0

            for chunk in pd.read_csv(
                    file_path, sep='\t', header=None, names=column_names,
                    usecols=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'],
                    chunksize=self.chunk_size, dtype='string', na_filter=False
            ):
                chunk_count += 1
                chunk_size = len(chunk)
                total_rows += chunk_size

                print(f"  - 分块 {chunk_count}: 读取 {chunk_size:,} 行, 累计 {total_rows:,} 行", end='')

                # 类型转换
                chunk['user_id'] = pd.to_numeric(chunk['user_id'].str.strip(), errors='coerce')
                chunk['item_id'] = pd.to_numeric(chunk['item_id'].str.strip(), errors='coerce')
                chunk['behavior_type'] = pd.to_numeric(chunk['behavior_type'].str.strip(), errors='coerce')
                chunk['item_category'] = pd.to_numeric(chunk['item_category'].str.strip(), errors='coerce')
                chunk['time'] = pd.to_datetime(chunk['time'].str.strip(), format='%Y-%m-%d %H', errors='coerce')

                chunk = chunk.dropna().copy()
                chunk = chunk[(chunk['behavior_type'] >= 1) & (chunk['behavior_type'] <= 4)]
                chunk = chunk[chunk['item_id'].isin(item_ids)]

                valid_size = len(chunk)
                processed_rows += valid_size
                print(f", 有效 {valid_size:,} 行")

                if valid_size == 0:
                    continue

                chunk['user_id'] = chunk['user_id'].astype(np.int64)
                chunk['item_id'] = chunk['item_id'].astype(np.int64)
                chunk['behavior_type'] = chunk['behavior_type'].astype(np.int8)
                chunk['item_category'] = chunk['item_category'].astype(np.int64)
                chunk['date'] = chunk['time'].dt.date

                # 按日期和用户商品对聚合行为
                for (date, uid, iid, btype), cnt in chunk.groupby(
                        ['date', 'user_id', 'item_id', 'behavior_type']).size().items():
                    daily_behaviors[date][(uid, iid)][btype - 1] += cnt

                # 统计用户活跃度
                for uid, group in chunk.groupby('user_id'):
                    user_activity[uid]['total'] += len(group)
                    user_activity[uid]['buy'] += (group['behavior_type'] == 4).sum()
                    user_activity[uid]['cart'] += (group['behavior_type'] == 3).sum()
                    user_activity[uid]['fav'] += (group['behavior_type'] == 2).sum()

                # 统计商品热度
                for iid, group in chunk.groupby('item_id'):
                    item_popularity[iid]['total'] += len(group)
                    item_popularity[iid]['buy'] += (group['behavior_type'] == 4).sum()
                    item_popularity[iid]['cart'] += (group['behavior_type'] == 3).sum()
                    item_popularity[iid]['fav'] += (group['behavior_type'] == 2).sum()

                # 统计类目
                for cat, group in chunk.groupby('item_category'):
                    category_stats[cat]['total'] += len(group)
                    category_stats[cat]['buy'] += (group['behavior_type'] == 4).sum()

                if total_rows >= self.max_rows:
                    print(f"\n  已达到最大行数限制 {self.max_rows:,}, 停止读取")
                    break

            if total_rows >= self.max_rows:
                break

        print(f"\n加载完成: 总读取 {total_rows:,} 行, 处理有效 {processed_rows:,} 行")
        print(f"覆盖日期数: {len(daily_behaviors)} 天")
        print(f"活跃用户数: {len(user_activity):,}")
        print(f"热门商品数: {len(item_popularity):,}")

        return daily_behaviors, user_activity, item_popularity, category_stats

    def extract_features_for_span(self, daily_behaviors, user_activity, item_popularity,
                                  category_stats, item_category_map, span):
        """
        提取增强特征(12维)

        特征:
        1-4: [浏览,收藏,加购,购买]次数
        5: 用户活跃度(总行为数)
        6: 用户购买率
        7: 商品热度(总行为数)
        8: 商品购买率
        9: 商品加购率
        10: 类目购买率
        11: 时间衰减权重
        12: 用户对该类目偏好度
        """
        features = []
        labels = []

        current_date = self.start_date.date()
        date_count = 0

        while current_date + timedelta(days=span) <= self.end_date.date():
            target_date = current_date + timedelta(days=span)

            if current_date not in daily_behaviors:
                current_date += timedelta(days=1)
                continue

            day_data = daily_behaviors[current_date]
            target_data = daily_behaviors.get(target_date, {})

            date_count += 1

            # 时间衰减: 距离预测日越近权重越高
            days_to_target = (self.target_date.date() - current_date).days
            time_decay = np.exp(-days_to_target / 7.0)

            for (uid, iid), behavior in day_data.items():
                # 基础特征(1-4维)
                feat = list(behavior)

                # 用户特征(5-6维)
                user_total = user_activity[uid]['total']
                user_buy_rate = user_activity[uid]['buy'] / user_total if user_total > 0 else 0
                feat.extend([user_total, user_buy_rate])

                # 商品特征(7-9维)
                item_total = item_popularity[iid]['total']
                item_buy_rate = item_popularity[iid]['buy'] / item_total if item_total > 0 else 0
                item_cart_rate = item_popularity[iid]['cart'] / item_total if item_total > 0 else 0
                feat.extend([item_total, item_buy_rate, item_cart_rate])

                # 类目特征(10维)
                cat = item_category_map.get(iid, -1)
                cat_buy_rate = 0
                if cat in category_stats and category_stats[cat]['total'] > 0:
                    cat_buy_rate = category_stats[cat]['buy'] / category_stats[cat]['total']
                feat.append(cat_buy_rate)

                # 时间衰减(11维)
                feat.append(time_decay)

                # 用户-类目偏好(12维)
                user_cat_pref = 0
                if cat != -1:
                    user_cat_behaviors = sum(
                        1 for d in daily_behaviors.values()
                        for (u, i) in d.keys()
                        if u == uid and item_category_map.get(i, -1) == cat
                    )
                    user_cat_pref = user_cat_behaviors / user_total if user_total > 0 else 0
                feat.append(user_cat_pref)

                features.append((uid, iid, feat))

                # 标签: span天后是否购买
                label = 1 if target_data.get((uid, iid), [0, 0, 0, 0])[3] > 0 else 0
                labels.append(label)

            if date_count % 5 == 0:
                print(f"    处理日期 {current_date}, 已提取 {len(features):,} 个样本")

            current_date += timedelta(days=1)

        weight = sum(labels) / len(labels) if labels else 0.0
        return features, labels, weight

    def train_models(self, daily_behaviors, user_activity, item_popularity,
                     category_stats, item_category_map):
        """训练多个模型(LR + RF + GBDT)"""
        print("\n训练分类器...")
        models = []
        weights = []

        for span in range(1, self.max_span + 1):
            print(f"\n[时间窗口 {span} 天]")
            features, labels, weight = self.extract_features_for_span(
                daily_behaviors, user_activity, item_popularity,
                category_stats, item_category_map, span
            )

            if len(features) == 0:
                print("  无可用样本,跳过")
                continue

            print(f"  提取样本数: {len(features):,}")
            print(f"  正样本比例: {weight:.4f}")

            # 特征矩阵(12维)
            X = np.array([feat[2] for feat in features])
            y = np.array(labels)

            # 训练3个模型
            print("  训练逻辑回归...")
            lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
            lr_model.fit(X, y)

            print("  训练随机森林...")
            rf_model = RandomForestClassifier(n_estimators=50, max_depth=10,
                                              class_weight='balanced', random_state=42)
            rf_model.fit(X, y)

            print("  训练GBDT...")
            gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=5,
                                                  learning_rate=0.1, random_state=42)
            gb_model.fit(X, y)

            models.append({
                'lr': lr_model,
                'rf': rf_model,
                'gb': gb_model,
                'span': span,
                'features': features
            })
            weights.append(weight)

            print(f"  训练完成")

        # 归一化权重
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]

        print(f"\n训练完成: 共 {len(models)} 个时间窗口")
        print(f"权重分布: {[f'{w:.3f}' for w in weights]}")

        return models, weights

    def generate_predictions(self, models, weights, daily_behaviors, user_activity,
                             item_popularity, category_stats, item_category_map, top_n=30):
        """生成预测结果,融合3个模型"""
        print("\n生成预测...")

        user_item_scores = defaultdict(float)

        for model_info, weight in zip(models, weights):
            lr_model = model_info['lr']
            rf_model = model_info['rf']
            gb_model = model_info['gb']
            span = model_info['span']

            feature_date = (self.target_date - timedelta(days=span)).date()

            if feature_date not in daily_behaviors:
                print(f"  时间窗口 {span} 天: 无数据")
                continue

            day_data = daily_behaviors[feature_date]
            print(f"  时间窗口 {span} 天: {len(day_data):,} 个用户商品对", end='')

            processed = 0
            days_to_target = (self.target_date.date() - feature_date).days
            time_decay = np.exp(-days_to_target / 7.0)

            for (uid, iid), behavior in day_data.items():
                # 构建12维特征
                feat = list(behavior)

                user_total = user_activity[uid]['total']
                user_buy_rate = user_activity[uid]['buy'] / user_total if user_total > 0 else 0
                feat.extend([user_total, user_buy_rate])

                item_total = item_popularity[iid]['total']
                item_buy_rate = item_popularity[iid]['buy'] / item_total if item_total > 0 else 0
                item_cart_rate = item_popularity[iid]['cart'] / item_total if item_total > 0 else 0
                feat.extend([item_total, item_buy_rate, item_cart_rate])

                cat = item_category_map.get(iid, -1)
                cat_buy_rate = 0
                if cat in category_stats and category_stats[cat]['total'] > 0:
                    cat_buy_rate = category_stats[cat]['buy'] / category_stats[cat]['total']
                feat.append(cat_buy_rate)

                feat.append(time_decay)

                user_cat_pref = 0
                if cat != -1:
                    user_cat_behaviors = sum(
                        1 for d in daily_behaviors.values()
                        for (u, i) in d.keys()
                        if u == uid and item_category_map.get(i, -1) == cat
                    )
                    user_cat_pref = user_cat_behaviors / user_total if user_total > 0 else 0
                feat.append(user_cat_pref)

                X = np.array([feat])

                # 融合3个模型的预测概率
                lr_prob = lr_model.predict_proba(X)[0][1]
                rf_prob = rf_model.predict_proba(X)[0][1]
                gb_prob = gb_model.predict_proba(X)[0][1]

                # 加权平均(LR:0.3, RF:0.4, GBDT:0.3)
                final_prob = 0.3 * lr_prob + 0.4 * rf_prob + 0.3 * gb_prob

                user_item_scores[(uid, iid)] += final_prob * weight

                processed += 1
                if processed % 10000 == 0:
                    print(f"\r  时间窗口 {span} 天: 已处理 {processed:,}/{len(day_data):,}", end='')

            print(f"\r  时间窗口 {span} 天: 完成预测 {len(day_data):,} 个用户商品对")

        print(f"\n候选用户商品对: {len(user_item_scores):,}")

        # 过滤低质量推荐(得分阈值)
        print("应用得分阈值过滤...")
        score_threshold = np.percentile(list(user_item_scores.values()), 50)  # 取中位数
        user_item_scores = {k: v for k, v in user_item_scores.items() if v >= score_threshold}
        print(f"过滤后候选数: {len(user_item_scores):,}")

        # 按用户分组并取Top-N
        print("按用户分组并排序...")
        user_recommendations = defaultdict(list)
        for (uid, iid), score in user_item_scores.items():
            user_recommendations[uid].append((iid, score))

        print(f"总用户数: {len(user_recommendations):,}")

        recommendations = []
        user_count = 0
        for uid, items in user_recommendations.items():
            items.sort(key=lambda x: x[1], reverse=True)
            # 动态调整Top-N:高活跃用户推荐更多
            user_total = user_activity[uid]['total']
            dynamic_top_n = min(top_n, max(5, int(user_total / 10)))

            for iid, _ in items[:dynamic_top_n]:
                recommendations.append((uid, iid))

            user_count += 1
            if user_count % 1000 == 0:
                print(f"  已处理 {user_count:,}/{len(user_recommendations):,} 个用户", end='\r')

        print(f"\n生成推荐: {len(recommendations):,} 条")
        return recommendations

    def save_results(self, recommendations, out_dir=r"result"):
        """保存推荐结果"""
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = os.path.join(out_dir, f"recommendation_{timestamp}.txt")

        print(f"\n保存结果到: {out_file}")
        with open(out_file, 'w', encoding='utf-8') as f:
            for uid, iid in recommendations:
                f.write(f"{uid}\t{iid}\n")

        print(f"保存完成: {len(recommendations):,} 条推荐")
        user_count = len(set(uid for uid, _ in recommendations))
        print(f"推荐用户数: {user_count:,}")
        print(f"平均每用户推荐: {len(recommendations) / user_count:.2f} 个商品")
        return out_file


def main():
    rec_sys = TimeWindowRecommendationSystem(
        max_span=7,
        chunk_size=500_000,
        max_rows=50_000_000
    )

    item_ids, item_category_map = rec_sys.load_item_subset()
    daily_behaviors, user_activity, item_popularity, category_stats = rec_sys.load_behavior_data(item_ids)

    models, weights = rec_sys.train_models(
        daily_behaviors, user_activity, item_popularity,
        category_stats, item_category_map
    )

    recommendations = rec_sys.generate_predictions(
        models, weights, daily_behaviors, user_activity,
        item_popularity, category_stats, item_category_map, top_n=10
    )

    out_file = rec_sys.save_results(recommendations)

    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
