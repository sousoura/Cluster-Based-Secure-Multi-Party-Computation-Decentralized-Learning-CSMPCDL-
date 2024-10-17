# frontend_display.py

import matplotlib.pyplot as plt

def plot_errors(errors_dict):
    plt.figure(figsize=(16, 10))

    colors = {
        'Centralized': 'blue',
        'Individual': 'green',
        'Federated': 'red',
        'Distributed': 'purple',
        'Cluster Cooperative': 'orange'
    }

    linestyles = {
        'Centralized': '-',
        'Individual': '--',
        'Federated': '-.',
        'Distributed': ':',
        'Cluster Cooperative': '-'
    }

    for algo_name, errors in errors_dict.items():
        plt.plot(
            range(1, len(errors) + 1),
            errors,
            label=algo_name,
            color=colors.get(algo_name, 'black'),
            linestyle=linestyles.get(algo_name, '-'),
            linewidth=2,
            alpha=0.8
        )

    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average Error Rate', fontsize=14)
    # plt.title('Average Error Rate Comparison over 100 Runs', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig('learning_algorithms_comparison.png', dpi=300)
    plt.show()
