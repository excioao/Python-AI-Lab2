"""
任务二：逻辑斯谛分类 (Logistic Regression)
--------------------------------------------------------
使用表 1 的 20 条训练数据训练，对表 2 的 6 条测试数据预测
计算准确率、召回率、F1 分数
"""
import numpy as np
from data_utils import get_train_data, get_test_data, StandardScaler, evaluate


# ==================== 逻辑斯谛回归实现 ====================
class LogisticRegression:
    """逻辑斯谛回归 —— 梯度下降 + Sigmoid"""

    def __init__(self, learning_rate=0.1, max_iter=2000):
        self.lr = learning_rate
        self.max_iter = max_iter

    def _sigmoid(self, z):
        """Sigmoid 激活: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        训练模型（梯度下降最小化交叉熵损失）
        X: (m, n) 特征矩阵, y: (m,) 标签向量 (0/1)
        """
        m, n = X.shape
        self.w = np.zeros(n)            # 权重初始化为零
        self.b = 0.0                    # 偏置初始化为零

        for epoch in range(self.max_iter):
            z = np.dot(X, self.w) + self.b       # 线性组合
            y_pred = self._sigmoid(z)             # 通过 Sigmoid
            # 交叉熵损失的梯度
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        """返回正类概率 P(y=1 | x)"""
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        """以 threshold 为界输出 0/1 分类"""
        return (self.predict_proba(X) >= threshold).astype(int)


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 65)
    print("  任务二：逻辑斯谛回归分类")
    print("=" * 65)

    # 加载数据
    X_train, y_train = get_train_data()
    X_test,  y_test  = get_test_data()

    # Z-Score 标准化（逻辑斯谛回归对特征尺度敏感）
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)

    # 训练
    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train_norm, y_train)

    # 预测
    y_pred = model.predict(X_test_norm)
    y_prob = model.predict_proba(X_test_norm)

    # 输出参数
    feature_names = ['每平米单价', '面积', '离地铁距离', '房龄']
    print("\n  模型参数:")
    for name, wi in zip(feature_names, model.w):
        direction = "[UP] 好卖" if wi > 0 else "[DOWN] 不好卖"
        print(f"    {name}: w = {wi:+.4f}  ({direction})")
    print(f"    偏置 b = {model.b:.4f}")

    # 输出预测结果
    print("\n  测试集预测结果:")
    print(f"    {'编号':<6} {'真实':<6} {'预测':<6} {'概率':<8}")
    print(f"    {'-' * 26}")
    for i in range(len(y_test)):
        print(f"    {i+1:<6} {y_test[i]:<6} {y_pred[i]:<6} {y_prob[i]:.4f}")

    # 评价指标
    acc, rec, f1 = evaluate(y_test, y_pred)
    print(f"\n  评价指标:")
    print(f"    准确率 (Accuracy): {acc:.2%}")
    print(f"    召回率 (Recall):   {rec:.2%}")
    print(f"    F1 分数:           {f1:.2%}")

    print("\n  [*] 逻辑斯谛回归是一个线性分类器，在本数据集上表现良好。")
