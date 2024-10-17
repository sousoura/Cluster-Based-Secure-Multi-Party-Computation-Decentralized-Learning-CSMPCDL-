# main.py

import numpy as np
import centralized_learning
import individual_learning
import federated_learning
import distributed_learning
import cluster_cooperation_learning
import frontend_display

def main():
    num_runs = 100  # 实验次数
    algorithms = {
        'Centralized': centralized_learning.run_centralized_learning,
        'Individual': individual_learning.run_individual_learning,
        'Federated': federated_learning.run_federated_learning,
        'Distributed': distributed_learning.run_distributed_learning,
        'Cluster Cooperative': cluster_cooperation_learning.run_cluster_cooperative_learning
    }

    averaged_errors = {}

    for algo_name, algo_func in algorithms.items():
        print(f"Running {algo_name} Learning...")
        all_errors = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for {algo_name}")
            errors = algo_func()
            all_errors.append(errors)

        # 找到最长的误差列表长度
        max_length = max(len(errors) for errors in all_errors)

        # 初始化一个数组来存储累加的误差
        summed_errors = np.zeros(max_length)

        # 累加每一步的误差
        for errors in all_errors:
            # 如果误差列表长度不足，使用最后一个误差值进行填充
            if len(errors) < max_length:
                errors.extend([errors[-1]] * (max_length - len(errors)))
            summed_errors += np.array(errors)

        # 计算平均误差
        averaged_errors[algo_name] = (summed_errors / num_runs).tolist()

    # 绘制平均误差曲线
    frontend_display.plot_errors(averaged_errors)

if __name__ == "__main__":
    main()
