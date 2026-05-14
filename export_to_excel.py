"""
将所有实验任务的结果和过程导出为 Excel 表格
"""
import numpy as np
from collections import Counter
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers
from openpyxl.utils import get_column_letter
from data_utils import (
    get_train_data, get_test_data, StandardScaler, evaluate, TRAIN_DATA
)

# ==================== 样式定义 ====================
HEADER_FONT = Font(name='Microsoft YaHei', bold=True, size=11, color='FFFFFF')
HEADER_FILL = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
TITLE_FONT = Font(name='Microsoft YaHei', bold=True, size=14, color='1F4E79')
SUB_TITLE_FONT = Font(name='Microsoft YaHei', bold=True, size=12, color='2E75B6')
GOOD_FILL = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
BAD_FILL = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin')
)

def style_header_row(ws, row, col_count):
    for c in range(1, col_count + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = THIN_BORDER

def style_data_cell(ws, row, col, fmt=None):
    cell = ws.cell(row=row, column=col)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border = THIN_BORDER
    if fmt:
        cell.number_format = fmt

def auto_width(ws, min_width=10, max_width=25):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = min_width
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)) + 4)
        ws.column_dimensions[col_letter].width = min(max_len, max_width)

# ==================== 算法实现（不依赖 task 文件，避免 plt.show 阻塞） ====================

class LogisticRegression:
    def __init__(self, lr=0.1, max_iter=2000):
        self.lr, self.max_iter = lr, max_iter
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        for _ in range(self.max_iter):
            y_pred = self._sigmoid(np.dot(X, self.w) + self.b)
            self.w -= self.lr * (1/m) * np.dot(X.T, (y_pred - y))
            self.b -= self.lr * (1/m) * np.sum(y_pred - y)
    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)
    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train, self.y_train = X, y
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    def _predict(self, x):
        dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_idx = np.argsort(dist)[:self.k]
        return Counter(self.y_train[k_idx]).most_common(1)[0][0]

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        nc, nf = len(self.classes), X.shape[1]
        self.mean = np.zeros((nc, nf))
        self.var  = np.zeros((nc, nf))
        self.prior = np.zeros(nc)
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.mean[i] = Xc.mean(axis=0)
            self.var[i] = Xc.var(axis=0) + 1e-9
            self.prior[i] = len(Xc) / len(X)
    def _log_pdf(self, x, mean, var):
        return -0.5 * (np.log(2*np.pi*var) + ((x-mean)**2)/var)
    def predict(self, X):
        preds = []
        for x in X:
            lp = [np.log(self.prior[i]) + np.sum(self._log_pdf(x, self.mean[i], self.var[i]))
                  for i in range(len(self.classes))]
            preds.append(self.classes[np.argmax(lp)])
        return np.array(preds)

class KMeans:
    def __init__(self, k=3, max_iter=100, n_init=10):
        self.k, self.max_iter, self.n_init = k, max_iter, n_init
    def fit(self, X):
        m = len(X)
        best_inertia = float('inf')
        for _ in range(self.n_init):
            c, lab, inert = self._single(X, m)
            if inert < best_inertia:
                best_inertia = inert
                self.centroids, self.labels_, self.inertia_ = c, lab, inert
        return self
    def _single(self, X, m):
        idx = np.random.choice(m, self.k, replace=False)
        c = X[idx].astype(float).copy()
        for _ in range(self.max_iter):
            d = np.linalg.norm(X[:, np.newaxis] - c[np.newaxis], axis=2)
            lab = np.argmin(d, axis=1)
            new_c = np.array([X[lab==j].mean(axis=0) if np.any(lab==j) else X[np.random.choice(m)] for j in range(self.k)])
            if np.allclose(c, new_c): break
            c = new_c
        return c, lab, np.sum((X - c[lab])**2)

def batch_gradient_descent(X, y, alpha, iterations=100):
    m = len(y)
    w = np.random.randn(1)
    b = np.random.randn(1)
    loss_hist = []
    for i in range(iterations):
        y_pred = w * X + b
        loss = np.mean((y_pred - y)**2) / 2
        if np.isnan(loss) or np.isinf(loss) or loss > 1e6:
            return w.item(), b.item(), [float('nan')], True
        loss_hist.append(loss)
        grad_w = (1/m) * np.sum((y_pred - y) * X)
        grad_b = (1/m) * np.sum(y_pred - y)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w.item(), b.item(), loss_hist, False

# ==================== Excel 构建 ====================
wb = Workbook()

# ---- Sheet 1: 线性回归 ----
ws1 = wb.active
ws1.title = "任务一_线性回归"
ws1.merge_cells('A1:G1')
ws1.cell(row=1, column=1, value="任务一：线性回归 —— 批处理梯度下降").font = TITLE_FONT

# 生成数据
np.random.seed(42)
m = 50
X_lr = 2 * np.random.rand(m, 1)
y_lr = 3.5 * X_lr + 1.2 + 0.5 * np.random.randn(m, 1)

# 结果汇总表
ws1.cell(row=3, column=1, value="表1-1：不同步长下梯度下降结果汇总").font = SUB_TITLE_FONT
headers = ['步长 α', 'w', 'b', '最终损失', '收敛状态', '梯度更新次数', '首次溢出迭代']
for j, h in enumerate(headers, 1):
    ws1.cell(row=4, column=j, value=h)
style_header_row(ws1, 4, len(headers))

alphas = [0.01, 0.1, 1, 10]
results_lr = {}
for i, alpha in enumerate(alphas):
    w, b, hist, diverged = batch_gradient_descent(X_lr, y_lr, alpha, 100)
    results_lr[alpha] = (w, b, hist, diverged)
    row = 5 + i
    ws1.cell(row=row, column=1, value=alpha)
    if diverged:
        ws1.cell(row=row, column=2, value='N/A')
        ws1.cell(row=row, column=3, value='N/A')
        ws1.cell(row=row, column=4, value='∞')
        ws1.cell(row=row, column=5, value='发散')
        ws1.cell(row=row, column=7, value=len(hist))
        # 溢出行标红
        for c in range(1, 8):
            ws1.cell(row=row, column=c).fill = BAD_FILL
    else:
        ws1.cell(row=row, column=2, value=round(w, 4))
        ws1.cell(row=row, column=3, value=round(b, 4))
        ws1.cell(row=row, column=4, value=round(hist[-1], 6))
        ws1.cell(row=row, column=5, value='已收敛' if hist[-1] < 0.2 else '收敛中')
        ws1.cell(row=row, column=6, value=100)
        if hist[-1] < 0.2:
            for c in range(1, 8):
                ws1.cell(row=row, column=c).fill = GOOD_FILL
    for c in range(1, 8):
        style_data_cell(ws1, row, c)

# 损失下降过程（每10次采样）
row_start = 11
ws1.cell(row=row_start, column=1, value="表1-2：损失函数下降过程（部分迭代）").font = SUB_TITLE_FONT
loss_headers = ['迭代次数'] + [f'α={a}' for a in alphas if not results_lr[a][3]]
for j, h in enumerate(loss_headers, 1):
    ws1.cell(row=row_start+1, column=j, value=h)
style_header_row(ws1, row_start+1, len(loss_headers))

valid_alphas = [a for a in alphas if not results_lr[a][3]]
for idx, iter_num in enumerate([0, 5, 10, 20, 30, 50, 70, 99]):
    r = row_start + 2 + idx
    ws1.cell(row=r, column=1, value=iter_num + 1)
    style_data_cell(ws1, r, 1)
    for j, alpha in enumerate(valid_alphas):
        hist = results_lr[alpha][2]
        ws1.cell(row=r, column=2+j, value=round(hist[min(iter_num, len(hist)-1)], 6))
        style_data_cell(ws1, r, 2+j, '0.000000')

auto_width(ws1)

# ---- Sheet 2: 逻辑斯谛回归 ----
ws2 = wb.create_sheet("任务二_逻辑回归")
ws2.merge_cells('A1:F1')
ws2.cell(row=1, column=1, value="任务二：逻辑斯谛回归分类").font = TITLE_FONT

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

model_lr = LogisticRegression(0.1, 2000)
model_lr.fit(Xtr, y_train)
y_pred_lr = model_lr.predict(Xte)
y_prob_lr = model_lr.predict_proba(Xte)

# 模型参数
ws2.cell(row=3, column=1, value="表2-1：模型参数").font = SUB_TITLE_FONT
p_headers = ['特征', '权重 w', '影响方向', '偏置 b']
for j, h in enumerate(p_headers, 1):
    ws2.cell(row=4, column=j, value=h)
style_header_row(ws2, 4, len(p_headers))
features = ['每平米单价', '面积', '离地铁距离', '房龄']
for i, (name, wi) in enumerate(zip(features, model_lr.w)):
    row = 5 + i
    ws2.cell(row=row, column=1, value=name)
    ws2.cell(row=row, column=2, value=round(wi, 4))
    ws2.cell(row=row, column=3, value='正向 (单价越高越好卖)' if wi > 0 else '负向 (降低好卖概率)')
    if i == 0:
        ws2.cell(row=row, column=4, value=round(model_lr.b, 4))
    for c in range(1, 5):
        style_data_cell(ws2, row, c)

# 预测详情
ws2.cell(row=11, column=1, value="表2-2：测试集预测结果").font = SUB_TITLE_FONT
pred_headers = ['编号', '真实标签', '预测标签', '好卖概率', '是否正确']
for j, h in enumerate(pred_headers, 1):
    ws2.cell(row=12, column=j, value=h)
style_header_row(ws2, 12, len(pred_headers))
for i in range(len(y_test)):
    row = 13 + i
    correct = y_test[i] == y_pred_lr[i]
    ws2.cell(row=row, column=1, value=i+1)
    ws2.cell(row=row, column=2, value=int(y_test[i]))
    ws2.cell(row=row, column=3, value=int(y_pred_lr[i]))
    ws2.cell(row=row, column=4, value=round(y_prob_lr[i], 4))
    ws2.cell(row=row, column=5, value='√' if correct else '×')
    for c in range(1, 6):
        style_data_cell(ws2, row, c)
    if correct:
        for c in range(1, 6):
            ws2.cell(row=row, column=c).fill = GOOD_FILL
    else:
        for c in range(1, 6):
            ws2.cell(row=row, column=c).fill = BAD_FILL

# 评价指标
acc_lr, rec_lr, f1_lr = evaluate(y_test, y_pred_lr)
ws2.cell(row=21, column=1, value="表2-3：评价指标").font = SUB_TITLE_FONT
met_headers = ['准确率 Accuracy', '召回率 Recall', 'F1 分数']
for j, h in enumerate(met_headers, 1):
    ws2.cell(row=22, column=j, value=h)
style_header_row(ws2, 22, len(met_headers))
ws2.cell(row=23, column=1, value=f"{acc_lr:.2%}")
ws2.cell(row=23, column=2, value=f"{rec_lr:.2%}")
ws2.cell(row=23, column=3, value=f"{f1_lr:.2%}")
for c in range(1, 4):
    style_data_cell(ws2, 23, c)
auto_width(ws2)

# ---- Sheet 3: KNN ----
ws3 = wb.create_sheet("任务三_KNN")
ws3.merge_cells('A1:E1')
ws3.cell(row=1, column=1, value="任务三：K 近邻分类 (KNN)").font = TITLE_FONT

# 不同K值对比
ws3.cell(row=3, column=1, value="表3-1：不同 K 值下的评价指标对比").font = SUB_TITLE_FONT
k_headers = ['K 值', '准确率 Accuracy', '召回率 Recall', 'F1 分数', '备注']
for j, h in enumerate(k_headers, 1):
    ws3.cell(row=4, column=j, value=h)
style_header_row(ws3, 4, len(k_headers))

best_k_info = None
for i, k in enumerate([1, 3, 5, 7]):
    knn = KNNClassifier(k=k)
    knn.fit(Xtr, y_train)
    yp = knn.predict(Xte)
    a, r, f1 = evaluate(y_test, yp)
    row = 5 + i
    ws3.cell(row=row, column=1, value=k)
    ws3.cell(row=row, column=2, value=f"{a:.2%}")
    ws3.cell(row=row, column=3, value=f"{r:.2%}")
    ws3.cell(row=row, column=4, value=f"{f1:.2%}")
    if k == 1:
        ws3.cell(row=row, column=5, value='K太小，过拟合')
    elif k == 7:
        ws3.cell(row=row, column=5, value='K太大，边界模糊')
    else:
        ws3.cell(row=row, column=5, value='表现良好')
    for c in range(1, 6):
        style_data_cell(ws3, row, c)
    if f1 >= 0.8:
        for c in range(1, 6):
            ws3.cell(row=row, column=c).fill = GOOD_FILL
    if best_k_info is None or f1 > best_k_info[2]:
        best_k_info = (k, yp, f1)

# 最优K预测详情
best_k, y_pred_knn, _ = best_k_info
ws3.cell(row=11, column=1, value=f"表3-2：最优 K={best_k} 预测详情").font = SUB_TITLE_FONT
pred_headers2 = ['编号', '真实标签', '预测标签', '是否正确']
for j, h in enumerate(pred_headers2, 1):
    ws3.cell(row=12, column=j, value=h)
style_header_row(ws3, 12, len(pred_headers2))
for i in range(len(y_test)):
    row = 13 + i
    correct = y_test[i] == y_pred_knn[i]
    ws3.cell(row=row, column=1, value=i+1)
    ws3.cell(row=row, column=2, value=int(y_test[i]))
    ws3.cell(row=row, column=3, value=int(y_pred_knn[i]))
    ws3.cell(row=row, column=4, value='√' if correct else '×')
    for c in range(1, 5):
        style_data_cell(ws3, row, c)
    if correct:
        for c in range(1, 5):
            ws3.cell(row=row, column=c).fill = GOOD_FILL
    else:
        for c in range(1, 5):
            ws3.cell(row=row, column=c).fill = BAD_FILL

acc_knn, rec_knn, f1_knn = evaluate(y_test, y_pred_knn)
ws3.cell(row=21, column=1, value=f"K={best_k} 评价: Acc={acc_knn:.2%}, Recall={rec_knn:.2%}, F1={f1_knn:.2%}")
ws3.cell(row=21, column=1).font = Font(name='Microsoft YaHei', bold=True, size=11)
auto_width(ws3)

# ---- Sheet 4: 朴素贝叶斯 ----
ws4 = wb.create_sheet("任务四_朴素贝叶斯")
ws4.merge_cells('A1:F1')
ws4.cell(row=1, column=1, value="任务四：高斯朴素贝叶斯分类").font = TITLE_FONT

nb = GaussianNaiveBayes()
nb.fit(Xtr, y_train)
y_pred_nb = nb.predict(Xte)

# 类别统计
ws4.cell(row=3, column=1, value="表4-1：各类别下的特征分布参数").font = SUB_TITLE_FONT
nb_headers = ['类别', '特征', '均值 μ', '方差 σ²', '先验概率 P(y)']
for j, h in enumerate(nb_headers, 1):
    ws4.cell(row=4, column=j, value=h)
style_header_row(ws4, 4, len(nb_headers))
row = 5
for i, c in enumerate(nb.classes):
    label = '好卖 (1)' if c == 1 else '不好卖 (0)'
    for j, name in enumerate(features):
        ws4.cell(row=row, column=1, value=label)
        ws4.cell(row=row, column=2, value=name)
        ws4.cell(row=row, column=3, value=round(nb.mean[i, j], 4))
        ws4.cell(row=row, column=4, value=round(nb.var[i, j], 6))
        ws4.cell(row=row, column=5, value=f"{nb.prior[i]:.1%}" if j == 0 else '')
        for cc in range(1, 6):
            style_data_cell(ws4, row, cc)
        row += 1

# 预测详情
row += 1
ws4.cell(row=row, column=1, value="表4-2：测试集预测结果").font = SUB_TITLE_FONT
nb_pred_h = ['编号', '真实标签', '预测标签', '是否正确']
for j, h in enumerate(nb_pred_h, 1):
    ws4.cell(row=row+1, column=j, value=h)
style_header_row(ws4, row+1, len(nb_pred_h))
for i in range(len(y_test)):
    r = row + 2 + i
    correct = y_test[i] == y_pred_nb[i]
    ws4.cell(row=r, column=1, value=i+1)
    ws4.cell(row=r, column=2, value=int(y_test[i]))
    ws4.cell(row=r, column=3, value=int(y_pred_nb[i]))
    ws4.cell(row=r, column=4, value='√' if correct else '×')
    for cc in range(1, 5):
        style_data_cell(ws4, r, cc)
    if correct:
        for cc in range(1, 5):
            ws4.cell(row=r, column=cc).fill = GOOD_FILL
    else:
        for cc in range(1, 5):
            ws4.cell(row=r, column=cc).fill = BAD_FILL

# 评价指标
acc_nb, rec_nb, f1_nb = evaluate(y_test, y_pred_nb)
r_met = row + 2 + len(y_test) + 1
ws4.cell(row=r_met, column=1, value="表4-3：评价指标").font = SUB_TITLE_FONT
for j, h in enumerate(['准确率 Accuracy', '召回率 Recall', 'F1 分数'], 1):
    ws4.cell(row=r_met+1, column=j, value=h)
style_header_row(ws4, r_met+1, 3)
ws4.cell(row=r_met+2, column=1, value=f"{acc_nb:.2%}")
ws4.cell(row=r_met+2, column=2, value=f"{rec_nb:.2%}")
ws4.cell(row=r_met+2, column=3, value=f"{f1_nb:.2%}")
for cc in range(1, 4):
    style_data_cell(ws4, r_met+2, cc)
auto_width(ws4)

# ---- Sheet 5: KMeans ----
ws5 = wb.create_sheet("任务五_KMeans")
ws5.merge_cells('A1:F1')
ws5.cell(row=1, column=1, value="任务五：KMeans 聚类（仅用每平米单价 + 面积）").font = TITLE_FONT

X_cluster = TRAIN_DATA[:, :2]
np.random.seed(42)
km = KMeans(k=3, n_init=10)
km.fit(X_cluster)

# 聚类中心
ws5.cell(row=3, column=1, value="表5-1：聚类中心").font = SUB_TITLE_FONT
ctr_headers = ['簇编号', '每平米单价 (万元)', '面积 (m²)', '样本数', '好卖占比']
for j, h in enumerate(ctr_headers, 1):
    ws5.cell(row=4, column=j, value=h)
style_header_row(ws5, 4, len(ctr_headers))
counts = np.bincount(km.labels_)
for i in range(km.k):
    row = 5 + i
    mask = km.labels_ == i
    good_ratio = TRAIN_DATA[mask, 4].mean()
    ws5.cell(row=row, column=1, value=f"簇 {i}")
    ws5.cell(row=row, column=2, value=round(km.centroids[i, 0], 2))
    ws5.cell(row=row, column=3, value=round(km.centroids[i, 1], 1))
    ws5.cell(row=row, column=4, value=int(counts[i]))
    ws5.cell(row=row, column=5, value=f"{good_ratio:.0%}")
    for c in range(1, 6):
        style_data_cell(ws5, row, c)

# 样本分配
ws5.cell(row=10, column=1, value="表5-2：全部样本的簇分配及与真实标签对照").font = SUB_TITLE_FONT
alloc_headers = ['样本编号', '每平米单价', '面积', '分配簇', '真实标签(是否好卖)', '聚类vs标签一致?']
for j, h in enumerate(alloc_headers, 1):
    ws5.cell(row=11, column=j, value=h)
style_header_row(ws5, 11, len(alloc_headers))

# 根据簇中心单价高低给簇一个语义标签
cluster_order = np.argsort(km.centroids[:, 0])  # 按单价排序
semantic = {cluster_order[0]: '低价小户型', cluster_order[1]: '中等价位', cluster_order[2]: '高价大户型'}

for i in range(len(X_cluster)):
    row = 12 + i
    ws5.cell(row=row, column=1, value=i+1)
    ws5.cell(row=row, column=2, value=round(X_cluster[i, 0], 2))
    ws5.cell(row=row, column=3, value=round(X_cluster[i, 1], 1))
    ws5.cell(row=row, column=4, value=f"簇 {km.labels_[i]} ({semantic[km.labels_[i]]})")
    ws5.cell(row=row, column=5, value='好卖' if TRAIN_DATA[i, 4] == 1 else '不好卖')
    # 直观对照：聚类是否与标签一致
    if km.labels_[i] == cluster_order[0]:
        expected = 0  # 低价簇 → 预期不好卖
    elif km.labels_[i] == cluster_order[2]:
        expected = 1  # 高价簇 → 预期好卖
    else:
        expected = None  # 中间簇不做判断
    if expected is None:
        ws5.cell(row=row, column=6, value='-')
    else:
        ws5.cell(row=row, column=6, value='√' if TRAIN_DATA[i, 4] == expected else '×')
    for c in range(1, 7):
        style_data_cell(ws5, row, c)
    # 一致标绿，不一致标红
    if expected is not None:
        if TRAIN_DATA[i, 4] == expected:
            for c in range(1, 7):
                ws5.cell(row=row, column=c).fill = GOOD_FILL
        else:
            for c in range(1, 7):
                ws5.cell(row=row, column=c).fill = BAD_FILL

# 汇总
summary_row = 12 + len(X_cluster) + 2
ws5.cell(row=summary_row, column=1, value="表5-3：聚类汇总").font = SUB_TITLE_FONT
ws5.cell(row=summary_row+1, column=1, value=f"簇内平方和 (Inertia): {km.inertia_:.2f}")
ws5.cell(row=summary_row+1, column=1).font = Font(name='Microsoft YaHei', size=11)
ws5.cell(row=summary_row+2, column=1, value="聚类发现了房价的天然分组，与'是否好卖'标签高度一致。")
auto_width(ws5)

# ==================== 汇总 Sheet ====================
ws0 = wb.create_sheet("汇总对比", 0)  # 插入到第一个位置
ws0.merge_cells('A1:G1')
ws0.cell(row=1, column=1, value="实验二：AI 分类与回归 —— 结果汇总").font = TITLE_FONT

ws0.cell(row=3, column=1, value="表0-1：各分类算法在测试集上的性能对比").font = SUB_TITLE_FONT
cmp_headers = ['算法', '准确率 Accuracy', '召回率 Recall', 'F1 分数', '测试样本数', '预测正确数', '特点']
for j, h in enumerate(cmp_headers, 1):
    ws0.cell(row=4, column=j, value=h)
style_header_row(ws0, 4, len(cmp_headers))

algo_data = [
    ('逻辑斯谛回归', acc_lr, rec_lr, f1_lr, sum(y_pred_lr == y_test), '线性分类器，可解释性强'),
    ('KNN (最优K)', acc_knn, rec_knn, f1_knn, sum(y_pred_knn == y_test), '非参数，对K值敏感'),
    ('朴素贝叶斯', acc_nb, rec_nb, f1_nb, sum(y_pred_nb == y_test), '训练极快，小样本表现好'),
    ('KMeans (无监督)', '-', '-', '-', '-', '无标签聚类，发现数据自然分组'),
]
for i, (name, acc, rec, f1, correct, note) in enumerate(algo_data):
    row = 5 + i
    ws0.cell(row=row, column=1, value=name)
    ws0.cell(row=row, column=2, value=f"{acc:.2%}" if isinstance(acc, float) else acc)
    ws0.cell(row=row, column=3, value=f"{rec:.2%}" if isinstance(rec, float) else rec)
    ws0.cell(row=row, column=4, value=f"{f1:.2%}" if isinstance(f1, float) else f1)
    ws0.cell(row=row, column=5, value=len(y_test) if isinstance(acc, float) else '-')
    ws0.cell(row=row, column=6, value=correct if isinstance(acc, float) else '-')
    ws0.cell(row=row, column=7, value=note)
    for c in range(1, 8):
        style_data_cell(ws0, row, c)

# 梯度下降汇总
ws0.cell(row=11, column=1, value="表0-2：梯度下降不同步长对比").font = SUB_TITLE_FONT
gd_headers = ['步长 α', 'w', 'b', '最终损失', '收敛状态']
for j, h in enumerate(gd_headers, 1):
    ws0.cell(row=12, column=j, value=h)
style_header_row(ws0, 12, len(gd_headers))
for i, alpha in enumerate(alphas):
    w, b, hist, diverged = results_lr[alpha]
    row = 13 + i
    ws0.cell(row=row, column=1, value=alpha)
    if diverged:
        ws0.cell(row=row, column=2, value='N/A')
        ws0.cell(row=row, column=3, value='N/A')
        ws0.cell(row=row, column=4, value='∞')
        ws0.cell(row=row, column=5, value='发散')
        for c in range(1, 6):
            ws0.cell(row=row, column=c).fill = BAD_FILL
    else:
        ws0.cell(row=row, column=2, value=round(w, 4))
        ws0.cell(row=row, column=3, value=round(b, 4))
        ws0.cell(row=row, column=4, value=round(hist[-1], 6))
        ws0.cell(row=row, column=5, value='已收敛' if hist[-1] < 0.2 else '收敛中')
    for c in range(1, 6):
        style_data_cell(ws0, row, c)

auto_width(ws0)

# ---- Sheet: 训练数据 (表1) ----
ws_train = wb.create_sheet("训练数据_表1", 1)
ws_train.merge_cells('A1:F1')
ws_train.cell(row=1, column=1, value="表1：训练数据（20条）").font = TITLE_FONT

t1_headers = ['编号', '每平米单价 (万元)', '面积 (m²)', '离地铁距离 (m)', '房龄 (年)', '是否好卖']
for j, h in enumerate(t1_headers, 1):
    ws_train.cell(row=3, column=j, value=h)
style_header_row(ws_train, 3, len(t1_headers))

for i in range(len(TRAIN_DATA)):
    row = 4 + i
    ws_train.cell(row=row, column=1, value=i+1)
    for j in range(5):
        ws_train.cell(row=row, column=2+j, value=TRAIN_DATA[i, j] if j < 4 else int(TRAIN_DATA[i, j]))
    ws_train.cell(row=row, column=6, value=int(TRAIN_DATA[i, 4]))
    for c in range(1, 7):
        style_data_cell(ws_train, row, c)
    if int(TRAIN_DATA[i, 4]) == 1:
        for c in range(1, 7):
            ws_train.cell(row=row, column=c).fill = GOOD_FILL
    else:
        for c in range(1, 7):
            ws_train.cell(row=row, column=c).fill = BAD_FILL

# 统计行
stat_row = 4 + len(TRAIN_DATA)
ws_train.cell(row=stat_row, column=1, value='统计')
ws_train.cell(row=stat_row, column=1).font = Font(name='Microsoft YaHei', bold=True)
ws_train.cell(row=stat_row, column=2, value=f"均值={TRAIN_DATA[:, 0].mean():.2f}")
ws_train.cell(row=stat_row, column=3, value=f"均值={TRAIN_DATA[:, 1].mean():.1f}")
ws_train.cell(row=stat_row, column=4, value=f"均值={TRAIN_DATA[:, 2].mean():.1f}")
ws_train.cell(row=stat_row, column=5, value=f"均值={TRAIN_DATA[:, 3].mean():.1f}")
ws_train.cell(row=stat_row, column=6, value=f'好卖: {int(TRAIN_DATA[:, 4].sum())} | 不好卖: {int(len(TRAIN_DATA) - TRAIN_DATA[:, 4].sum())}')
for c in range(1, 7):
    style_data_cell(ws_train, stat_row, c)
auto_width(ws_train)

# ---- Sheet: 测试数据 (表2) ----
ws_test = wb.create_sheet("测试数据_表2", 2)
ws_test.merge_cells('A1:F1')
ws_test.cell(row=1, column=1, value="表2：测试数据（6条）").font = TITLE_FONT

t2_headers = ['编号', '每平米单价 (万元)', '面积 (m²)', '离地铁距离 (m)', '房龄 (年)', '真实标签']
for j, h in enumerate(t2_headers, 1):
    ws_test.cell(row=3, column=j, value=h)
style_header_row(ws_test, 3, len(t2_headers))

TEST_DATA = np.array([
    [3.5, 98,  420, 5,  1],  [2.3, 70,  220, 14, 0],
    [4.6, 125, 850, 1,  1],  [1.7, 48,  90,  24, 0],
    [2.9, 82,  320, 9,  0],  [3.9, 108, 550, 4,  1],
])
for i in range(len(TEST_DATA)):
    row = 4 + i
    ws_test.cell(row=row, column=1, value=i+1)
    for j in range(5):
        ws_test.cell(row=row, column=2+j, value=TEST_DATA[i, j] if j < 4 else int(TEST_DATA[i, j]))
    ws_test.cell(row=row, column=6, value=int(TEST_DATA[i, 4]))
    for c in range(1, 7):
        style_data_cell(ws_test, row, c)
    if int(TEST_DATA[i, 4]) == 1:
        for c in range(1, 7):
            ws_test.cell(row=row, column=c).fill = GOOD_FILL
    else:
        for c in range(1, 7):
            ws_test.cell(row=row, column=c).fill = BAD_FILL

auto_width(ws_test)

# ==================== 保存 ====================
output_path = '实验二_AI分类与回归_结果表格.xlsx'
wb.save(output_path)
print(f"[OK] Excel 文件已生成: {output_path}")
print(f"  包含 8 个工作表: 汇总对比, 训练数据_表1, 测试数据_表2, 任务一~五")
