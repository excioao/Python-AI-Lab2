"""
共享数据模块 —— 提供训练/测试数据、标准化、评价指标
实验二：AI 分类与回归
"""
import numpy as np

# ==================== 训练数据（表 1，20 条） ====================
# 特征: [每平米单价(万元), 面积(m²), 离地铁距离(m), 房龄(年), 是否好卖]
TRAIN_DATA = np.array([
    [3.2, 89,  300,  5,  1],  [2.8, 75,  150,  8,  1],
    [4.5, 120, 800,  2,  1],  [1.9, 55,  50,   20, 0],
    [3.0, 95,  400,  6,  1],  [2.2, 68,  200,  15, 0],
    [3.8, 105, 600,  3,  1],  [2.5, 80,  350,  10, 0],
    [4.0, 130, 1000, 1,  1],  [1.5, 45,  100,  25, 0],
    [3.6, 100, 500,  4,  1],  [2.0, 60,  250,  18, 0],
    [4.2, 115, 700,  2,  1],  [2.7, 85,  300,  12, 0],
    [1.8, 50,  80,   22, 0],  [3.3, 92,  450,  7,  1],
    [4.8, 140, 900,  0,  1],  [2.1, 62,  180,  16, 0],
    [3.1, 88,  380,  8,  0],  [4.1, 110, 650,  3,  1],
])

# ==================== 测试数据（表 2，6 条） ====================
TEST_DATA = np.array([
    [3.5, 98,  420, 5,  1],  [2.3, 70,  220, 14, 0],
    [4.6, 125, 850, 1,  1],  [1.7, 48,  90,  24, 0],
    [2.9, 82,  320, 9,  0],  [3.9, 108, 550, 4,  1],
])

# ==================== 获取训练/测试集 ====================
def get_train_data():
    """返回 X_train (20x4), y_train (20,)"""
    return TRAIN_DATA[:, :4], TRAIN_DATA[:, 4]

def get_test_data():
    """返回 X_test (6x4), y_test (6,)"""
    return TEST_DATA[:, :4], TEST_DATA[:, 4]

# ==================== 特征标准化 ====================
class StandardScaler:
    """Z-Score 标准化: x' = (x - mean) / std"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ==================== 评价指标 ====================
def evaluate(y_true, y_pred):
    """
    计算二分类评价指标
    返回: (准确率, 召回率, F1 分数)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return acc, rec, f1
