"""
任务四：朴素贝叶斯分类 (Gaussian Naive Bayes)
--------------------------------------------------------
使用表 1 训练数据、表 2 测试数据
假设各特征在给定类别下服从高斯分布
"""
import numpy as np
from data_utils import get_train_data, get_test_data, StandardScaler, evaluate


# ==================== 高斯朴素贝叶斯实现 ====================
class GaussianNaiveBayes:
    """高斯朴素贝叶斯 —— 连续特征 + 对数概率防下溢"""

    def fit(self, X, y):
        """
        训练: 为每个类别估计每个特征的 μ 和 σ²
        X: (m, n) 特征矩阵, y: (m,) 标签向量 (0/1)
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var  = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)        # 先验概率 P(y=c_k)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]                     # 属于类别 c 的样本子集
            self.mean[idx] = X_c.mean(axis=0)   # 各类别各特征的均值 μ
            self.var[idx]  = X_c.var(axis=0) + 1e-9  # 方差 σ²（+ε 防除零）
            self.prior[idx] = len(X_c) / len(X) # 先验概率估计

    def _gaussian_log_pdf(self, x, mean, var):
        """
        高斯分布的对数概率密度
        使用对数避免多个小概率连乘导致下溢
        """
        return -0.5 * (
            np.log(2 * np.pi * var) + ((x - mean) ** 2) / var
        )

    def predict(self, X):
        """
        预测: 选择后验概率（对数形式）最大的类别
        P(y=c|x) ∝ P(y=c) * ∏ P(x_j | y=c)
        取对数: log P(y=c|x) ∝ log P(y=c) + Σ log P(x_j | y=c)
        """
        predictions = []
        for x in X:
            log_posteriors = []
            for idx in range(len(self.classes)):
                # 对数先验 + 对数似然之和
                log_prior = np.log(self.prior[idx])
                log_likelihood = np.sum(
                    self._gaussian_log_pdf(x, self.mean[idx], self.var[idx])
                )
                log_posteriors.append(log_prior + log_likelihood)
            predictions.append(self.classes[np.argmax(log_posteriors)])
        return np.array(predictions)


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 65)
    print("  任务四：朴素贝叶斯分类 (Gaussian Naive Bayes)")
    print("=" * 65)

    # 加载数据
    X_train, y_train = get_train_data()
    X_test,  y_test  = get_test_data()

    # 标准化使特征更接近高斯分布
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)

    # 训练
    nb = GaussianNaiveBayes()
    nb.fit(X_train_norm, y_train)

    # 输出类别统计
    feature_names = ['每平米单价', '面积', '离地铁距离', '房龄']
    print("\n  各类别特征统计:")
    for idx, c in enumerate(nb.classes):
        label_str = "好卖" if c == 1 else "不好卖"
        print(f"\n  类别「{label_str}」 (n={int(nb.prior[idx] * len(y_train))}):")
        for j, name in enumerate(feature_names):
            print(f"    {name}: μ={nb.mean[idx, j]:+.3f}, σ²={nb.var[idx, j]:.4f}")

    # 预测
    y_pred = nb.predict(X_test_norm)

    print(f"\n  预测详情:")
    print(f"    真实标签: {y_test.astype(int).tolist()}")
    print(f"    预测标签: {y_pred.tolist()}")

    # 评价
    acc, rec, f1 = evaluate(y_test, y_pred)
    print(f"\n  评价指标:")
    print(f"    准确率 (Accuracy): {acc:.2%}")
    print(f"    召回率 (Recall):   {rec:.2%}")
    print(f"    F1 分数:           {f1:.2%}")

    print("\n  [*] 朴素贝叶斯训练极快，小样本下常有不俗表现；")
    print("    但特征独立性假设不成立时，概率估计会偏倚。")
