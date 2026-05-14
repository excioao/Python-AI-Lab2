"""
任务三：K 近邻分类 (KNN)
--------------------------------------------------------
使用表 1 训练数据、表 2 测试数据
对比不同 K 值 (1, 3, 5, 7) 下的分类性能
"""
import numpy as np
from collections import Counter
from data_utils import get_train_data, get_test_data, StandardScaler, evaluate


# ==================== KNN 分类器实现 ====================
class KNNClassifier:
    """K 近邻分类器 —— 欧氏距离 + 多数投票"""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """KNN 不进行显式训练，仅存储数据"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """对每个测试样本进行预测"""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """
        单样本预测:
        1. 计算到所有训练样本的欧氏距离
        2. 取距离最近的 K 个邻居
        3. 由邻居标签多数投票决定类别
        """
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_nearest_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_idx]
        # 多数投票
        return Counter(k_nearest_labels).most_common(1)[0][0]


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 65)
    print("  任务三：K 近邻分类 (KNN)")
    print("=" * 65)

    # 加载数据
    X_train, y_train = get_train_data()
    X_test,  y_test  = get_test_data()

    # Z-Score 标准化 —— KNN 基于距离，必须消除量纲差异
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)

    # 测试不同 K 值
    print("\n  不同 K 值下的分类性能:")
    print(f"    {'K值':<8} {'准确率':<12} {'召回率':<12} {'F1分数':<12}")
    print(f"    {'-' * 44}")

    best_k, best_f1 = None, 0
    for k in [1, 3, 5, 7]:
        knn = KNNClassifier(k=k)
        knn.fit(X_train_norm, y_train)
        y_pred = knn.predict(X_test_norm)
        acc, rec, f1 = evaluate(y_test, y_pred)
        print(f"    K={k:<7} {acc:<12.2%} {rec:<12.2%} {f1:<12.2%}")
        if f1 > best_f1:
            best_k, best_f1 = k, f1

    # 最优 K 值下的详细预测
    print(f"\n  [*] 最优 K 值: {best_k} (F1={best_k:.2%})")

    knn_best = KNNClassifier(k=best_k)
    knn_best.fit(X_train_norm, y_train)
    y_pred_best = knn_best.predict(X_test_norm)

    print(f"\n  预测详情 (K={best_k}):")
    print(f"    真实标签: {y_test.astype(int).tolist()}")
    print(f"    预测标签: {y_pred_best.tolist()}")

    # 评价
    acc, rec, f1 = evaluate(y_test, y_pred_best)
    print(f"\n  最终评价指标: Acc={acc:.2%}, Recall={rec:.2%}, F1={f1:.2%}")

    print("\n  [*] K 值过小易过拟合，过大则边界模糊 -- 交叉验证选 K 是最佳实践。")
