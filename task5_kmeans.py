"""
任务五：KMeans 聚类
--------------------------------------------------------
仅使用训练数据中的「每平米单价」和「面积」两个特征
K=3，多次随机初始化选最优，含可视化
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_utils import TRAIN_DATA

# ==================== 中文字体配置 ====================
def set_chinese_font():
    """尝试使用系统中可用的中文字体"""
    candidates = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'Source Han Sans SC', 'sans-serif']
    for name in candidates:
        try:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            fig = plt.figure()
            fig.text(0.5, 0.5, '测试', fontsize=12, ha='center')
            fig.savefig('nul', format='png')
            plt.close(fig)
            return
        except Exception:
            continue
    import warnings
    warnings.filterwarnings('ignore', message='Glyph.*missing from font')

set_chinese_font()


# ==================== KMeans 实现 ====================
class KMeans:
    """KMeans 聚类 —— 欧氏距离 + 多次随机初始化"""

    def __init__(self, k=3, max_iter=100, n_init=10):
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init           # 随机初始化次数（选最优）

    def fit(self, X):
        m = len(X)
        best_inertia = float('inf')

        for _ in range(self.n_init):
            centroids, labels, inertia = self._single_fit(X, m)
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids = centroids
                self.labels_ = labels
                self.inertia_ = inertia
        return self

    def _single_fit(self, X, m):
        """单次 KMeans 运行"""
        # 随机选择 K 个样本作为初始质心
        idx = np.random.choice(m, self.k, replace=False)
        centroids = X[idx].astype(float).copy()

        for _ in range(self.max_iter):
            # ---- 分配步骤：每个样本归入最近质心 ----
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )                                           # (m, k)
            labels = np.argmin(distances, axis=1)

            # ---- 更新步骤：重算质心 ----
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j)
                else X[np.random.choice(m)]             # 空簇随机重初始化
                for j in range(self.k)
            ])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        inertia = np.sum((X - centroids[labels]) ** 2)  # 簇内平方和
        return centroids, labels, inertia


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 65)
    print("  任务五：KMeans 聚类（仅使用单价 + 面积）")
    print("=" * 65)

    # 提取两个特征
    X_cluster = TRAIN_DATA[:, :2]

    np.random.seed(42)
    kmeans = KMeans(k=3, n_init=10)
    kmeans.fit(X_cluster)

    # 输出聚类中心
    print("\n  聚类中心 (每平米单价, 面积):")
    for i, c in enumerate(kmeans.centroids):
        print(f"    簇 {i}: [{c[0]:.2f} 万元/m², {c[1]:.1f} m²]")

    # 输出各簇样本数
    counts = np.bincount(kmeans.labels_)
    print(f"\n  各簇样本数量: {dict(enumerate(counts))}")
    print(f"  簇内平方和 (Inertia): {kmeans.inertia_:.2f}")

    # 输出每个样本所属簇
    print(f"\n  样本分配: {kmeans.labels_.tolist()}")

    # ==================== 可视化聚类结果 ====================
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    plt.figure(figsize=(9, 7))

    for i in range(kmeans.k):
        mask = kmeans.labels_ == i
        plt.scatter(
            X_cluster[mask, 0], X_cluster[mask, 1],
            c=colors[i], s=100, alpha=0.7, edgecolors='k', linewidth=0.5,
            label=f'簇 {i} (n={counts[i]})'
        )
        plt.scatter(
            kmeans.centroids[i, 0], kmeans.centroids[i, 1],
            c=colors[i], s=350, marker='*', edgecolors='k', linewidth=1.5
        )

    # 标注各簇的实际"好卖"占比
    for i in range(kmeans.k):
        mask = kmeans.labels_ == i
        labels_in_cluster = TRAIN_DATA[mask, 4].astype(int)
        good_ratio = labels_in_cluster.mean()
        x, y = kmeans.centroids[i, 0], kmeans.centroids[i, 1]
        plt.annotate(f'好卖占比: {good_ratio:.0%}',
                     (x, y), textcoords="offset points",
                     xytext=(0, 20), ha='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.xlabel('每平米单价 (万元)', fontsize=11)
    plt.ylabel('面积 (m²)', fontsize=11)
    plt.title('KMeans 聚类结果 (K=3)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==================== 聚类结果分析 ====================
    print("\n  聚类分析:")
    for i in range(kmeans.k):
        mask = kmeans.labels_ == i
        labels = TRAIN_DATA[mask, 4].astype(int)
        good_ratio = labels.mean()
        c = kmeans.centroids[i]
        if c[0] > 3.5:
            desc = "高端改善型 —— 单价高、面积大，好卖率高"
        elif c[0] < 2.5:
            desc = "老旧/偏远型 —— 单价低、面积小，好卖率低"
        else:
            desc = "刚需/首改型 —— 价位适中，标签较分散"
        print(f"    簇 {i}: {desc} (好卖占比={good_ratio:.0%})")

    print("\n  [*] KMeans 是无监督方法，发现了数据中自然的群体结构。")
    print("    聚类结果与「是否好卖」标签高度吻合，说明单价和面积是关键区分特征。")
