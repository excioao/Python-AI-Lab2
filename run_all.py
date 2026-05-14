"""
实验二：AI 分类与回归 —— 全任务运行入口
依次执行 5 个任务，含图表的任务异步启动（窗口不阻塞后续任务）
"""
import subprocess
import sys

tasks = [
    ("任务一：线性回归（梯度下降）",       "task1_linear_regression.py"),
    ("任务二：逻辑斯谛回归",               "task2_logistic_regression.py"),
    ("任务三：K 近邻分类 (KNN)",           "task3_knn.py"),
    ("任务四：朴素贝叶斯分类",             "task4_naive_bayes.py"),
    ("任务五：KMeans 聚类",                "task5_kmeans.py"),
]

# 有图表窗口的任务 —— 后台启动，不等待
plot_scripts = {"task1_linear_regression.py", "task5_kmeans.py"}

print("=" * 60)
print("  实验二：人工智能导论 -- AI 分类与回归实践")
print("=" * 60)
print()

background_procs = []
all_passed = True

for task_name, script in tasks:
    print(f"{'─' * 60}")
    print(f"  >>> 开始: {task_name} ({script})")
    print(f"{'─' * 60}")

    if script in plot_scripts:
        # 图表任务：后台启动，窗口一直显示直到手动关闭
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        background_procs.append((task_name, proc))
        print(f"  [ASYNC] {task_name} 图表窗口已弹出，继续执行后续任务...\n")
    else:
        # 无图表任务：同步执行，显示输出
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  [FAIL] {task_name} 返回码={result.returncode}")
            all_passed = False
        else:
            print(f"  [OK] {task_name} 完成。\n")

print(f"\n{'=' * 60}")
if all_passed:
    print("  全部 5 个任务执行完毕!")
else:
    print("  部分任务执行异常，请查看上方日志。")

if background_procs:
    names = [n for n, _ in background_procs]
    print(f"  图表窗口 ({', '.join(names)}) 仍在显示中，关闭后程序退出。")
print(f"{'=' * 60}")
