import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    def __init__(self):
        self.records = []

    def record(self, info):
        self.records.append(info)

    def save(self, path):
        pd.DataFrame(self.records).to_csv(path, index=False)

    def plot(self):
        df = pd.DataFrame(self.records)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        df['latency'].expanding().mean().plot(ax=axs[0], title='Avg Latency')
        df['throughput'].expanding().mean().plot(ax=axs[1], title='Avg Throughput')
        df['orphan_rate'].expanding().mean().plot(ax=axs[2], title='Avg Orphan Rate')
        plt.tight_layout()
        plt.savefig('metrics.png')

class ComparativeMetrics:
    def __init__(self):
        self.data = {}

    def add(self, name, metrics_list):
        self.data[name] = metrics_list

    def plot_comparison(self):
        names = list(self.data.keys())
        avg_latency = [np.mean(self.data[n]['latency']) for n in names]
        avg_throughput = [np.mean(self.data[n]['throughput']) for n in names]
        avg_orphan = [np.mean(self.data[n]['orphan_rate']) for n in names]
        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, avg_latency, width, label='Latency')
        ax.bar(x, avg_throughput, width, label='Throughput')
        ax.bar(x + width, avg_orphan, width, label='Orphan Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Average Value')
        ax.set_title('Policy Comparison')
        ax.legend()
        plt.tight_layout()
        plt.savefig('comparison_bar.png')
