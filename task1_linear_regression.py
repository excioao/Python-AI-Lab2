"""
任务一：线性回归 —— 批处理梯度下降拟合
--------------------------------------------------------
对一维数据 y = 3.5x + 1.2 + noise 进行拟合
对比步长 α ∈ {0.01, 0.1, 1, 10} 在 100 次迭代下的收敛行为
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 兼容 PyCharm 科学模式
import matplotlib.pyplot as plt

# ==================== 中文字体配置 ====================
def set_chinese_font():
    """尝试使用系统中可用的中文字体，避免图表中文乱码/缺字"""
    from matplotlib.font_manager import FontProperties
    candidates = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'Source Han Sans SC', 'sans-serif']
    for name in candidates:
        try:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            # 验证字体是否真的可用
            fig = plt.figure()
            fig.text(0.5, 0.5, '测试', fontsize=12, ha='center')
            fig.savefig('nul', format='png')  # Windows: NUL, Unix: /dev/null
            plt.close(fig)
            return
        except Exception:
            continue
    # 回退：所有中文字体都不可用时，避免 warning 刷屏
    import warnings
    warnings.filterwarnings('ignore', message='Glyph.*missing from font')

set_chinese_font()


# ==================== 生成模拟数据 ====================
np.random.seed(42)
m = 50                                  # 样本数量
X = 2 * np.random.rand(m, 1)            # 在 [0, 2) 上均匀采样
true_w, true_b = 3.5, 1.2               # 真实参数
noise = 0.5 * np.random.randn(m, 1)     # 高斯噪声
y = true_w * X + true_b + noise


# ==================== 批处理梯度下降 ====================
def batch_gradient_descent(X, y, alpha, iterations=100):
    """
    批处理梯度下降求解线性回归参数
    参数:
        X          : 特征矩阵 (m, 1)
        y          : 标签向量 (m, 1)
        alpha      : 学习率（步长）
        iterations : 最大迭代次数
    返回:
        w, b          : 训练得到的参数
        loss_history  : 每次迭代的损失值列表
        diverged      : 是否发散
    """
    m = len(y)
    w = np.random.randn(1)              # 随机初始化权重
    b = np.random.randn(1)              # 随机初始化偏置
    loss_history = []

    for i in range(iterations):
        y_pred = w * X + b              # 前向传播
        loss = np.mean((y_pred - y) ** 2) / 2  # MSE / 2

        # 发散检测：损失变为 NaN/Inf，或超出合理范围
        if np.isnan(loss) or np.isinf(loss) or loss > 1e6:
            loss_history.append(float('nan'))
            return w.item(), b.item(), loss_history, True

        loss_history.append(loss)

        # 计算梯度
        grad_w = (1 / m) * np.sum((y_pred - y) * X)
        grad_b = (1 / m) * np.sum(y_pred - y)

        # 参数更新
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w.item(), b.item(), loss_history, False


# ==================== 不同步长训练 ====================
alphas = [0.01, 0.1, 1, 10]
results = {}

print("=" * 65)
print("  任务一：线性回归 —— 批处理梯度下降")
print("=" * 65)
print(f"  真实参数: w={true_w}, b={true_b}\n")

for alpha in alphas:
    w, b, hist, diverged = batch_gradient_descent(X, y, alpha, iterations=100)
    results[alpha] = {'w': w, 'b': b, 'loss': hist, 'diverged': diverged}
    if diverged:
        print(f"  步长 alpha={alpha:<6} | [FAIL] 发散! 步长过大，参数爆炸 (损失在第 {len(hist)} 次迭代溢出)")
    else:
        final_loss = hist[-1]
        status = "[OK] 已收敛" if final_loss < 0.2 else "~ 收敛中"
        print(f"  步长 α={alpha:<6} | w={w:8.4f}  b={b:8.4f}  "
              f"最终损失={final_loss:.6f}  {status}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('任务一：不同步长下的梯度下降行为', fontsize=14, fontweight='bold')

# 左上：损失下降曲线对比（只画未发散的，α=10 用虚线标记）
ax_loss = axes[0, 0]
for alpha in alphas:
    loss_vals = results[alpha]['loss']
    if results[alpha]['diverged']:
        ax_loss.plot([0], [0], '--', color='gray', linewidth=1, label=f'α={alpha} (发散)')
    else:
        ax_loss.plot(loss_vals, label=f'α={alpha}', linewidth=1.5)
ax_loss.set_xlabel('迭代次数', fontsize=10)
ax_loss.set_ylabel('损失 (MSE/2)', fontsize=10)
ax_loss.set_title('损失函数下降曲线对比')
ax_loss.legend(fontsize=9)
ax_loss.grid(True, alpha=0.3)

# 其余三个子图：各步长拟合效果
for idx, alpha in enumerate(alphas[1:], start=1):
    ax = axes[idx // 2][idx % 2]
    w, b = results[alpha]['w'], results[alpha]['b']
    ax.scatter(X, y, alpha=0.5, s=18, label='训练数据')
    x_line = np.linspace(0, 2, 100).reshape(-1, 1)
    if results[alpha]['diverged']:
        ax.text(1, np.mean(y), '[FAIL] 发散\n参数过大无法绘制', ha='center', fontsize=14, color='red')
        ax.set_title(f'步长 α={alpha}  已发散')
    else:
        ax.plot(x_line, w * x_line + b, 'r-', linewidth=2,
                label=f'拟合: y={w:.2f}x+{b:.2f}')
        ax.set_title(f'步长 α={alpha} 拟合结果')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 第一个子图分别画四种步长的拟合线
ax_all = axes[0, 1]
ax_all.scatter(X, y, alpha=0.5, s=18, label='训练数据')
x_line = np.linspace(0, 2, 100).reshape(-1, 1)
for alpha in alphas[:3]:  # 只画 0.01, 0.1, 1（10 已发散）
    w, b = results[alpha]['w'], results[alpha]['b']
    ax_all.plot(x_line, w * x_line + b, linewidth=1.5,
                label=f'α={alpha}: y={w:.2f}x+{b:.2f}')
ax_all.set_xlabel('x')
ax_all.set_ylabel('y')
ax_all.set_title('各步长拟合直线对比（不含 α=10）')
ax_all.legend(fontsize=8)
ax_all.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n  [*] 结论: alpha=0.1 收敛最稳定; alpha=10 发散 -- 步长不宜过大。")
