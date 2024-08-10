import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_train_dev_metric(epochs, train_metric, eval_metric, base_path, metric_name, task_type):
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplots(figsize=(7, 6))
    
    colors = {'Loss': {'Train': '#00008B', 'Valid': '#228B22'}, 'AUC': {'Train': '#4B0082', 'Valid': '#FF8C00'}, \
                'PRC': {'Train': '#000080', 'Valid': '#ff69b4'}, 'RMSE': {'Train': '#000080', 'Valid': '#ff69b4'}, \
                'MAE': {'Train': '#000080', 'Valid': '#ff69b4'}, 'Pearson': {'Train': '#000080', 'Valid': '#ff69b4'}, \
                'Spearman': {'Train': '#000080', 'Valid': '#ff69b4'}, 'Accuracy': {'Train': '#000080', 'Valid': '#ff69b4'}}
    
    plt.plot(epochs, train_metric, color=colors[metric_name]['Train'], lw=2, label='Train')
    plt.plot(epochs, eval_metric, color=colors[metric_name]['Valid'], lw=2, label='Valid')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    current_yticks = plt.gca().get_yticks()
    for y in current_yticks:
        plt.axhline(y=y, color='grey', linestyle='-', lw=0.77, alpha=0.37)  # 根据y轴刻度添加水平刻度线
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel(metric_name, fontsize=13)
    plt.title('%s curves on fine-tuning of %s' % (metric_name.capitalize(), task_type.title()), fontsize=14)
    plt.legend(loc="lower right",fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, task_type + '_' + metric_name +'.jpg'), bbox_inches='tight', dpi=1024)
