import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy.stats.stats import DescribeResult

os.makedirs("images", exist_ok=True)
paths = ["originals"] + ["improved" + str(i) for i in range(1, 7)]
names = ["original", "Deep Normal", "Deep Sobel", "Sobel", "Normal", "Deep Laplacian", "Laplacian"]

permutation = [0, 4, 6, 3, 1, 5, 2]
permutated_names = [names[i] for i in permutation]
js = []
for p in paths:
    with open("stats/stats_" + p + ".json") as in_file:
        js.append(json.load(in_file))
means = [[s[2] for s in d.values()] for d in js]
permutated_means = [means[i] for i in permutation]
identities = js[0].keys()

y_pos = np.arange(len(paths))


def plot_all():
    for (t, id) in zip(zip(*permutated_means), identities):
        fig, ax = plt.subplots(figsize=(10, 10))
        bar = plt.bar(y_pos, list(t), align='center', alpha=0.5)
        plt.xticks(y_pos, permutated_names)
        plt.ylabel('Mean')
        plt.title(id)
        for rect, m in zip(bar, list(t)):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"{m:.4f}", ha='center', va='bottom')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        plt.tight_layout()
        plt.savefig(f"images/{id}.png")
        plt.close(fig)


# plot_all()

total_stats: List[DescribeResult] = [stats.describe(m) for m in permutated_means]
print(total_stats)
fig, ax = plt.subplots(figsize=(10, 10))
bar = plt.bar(y_pos, [m.mean for m in total_stats],
              align='center',
              alpha=0.5,
              yerr=list(zip(*[m.minmax for m in total_stats])),
              ecolor='black',
              capsize=10)
plt.xticks(y_pos, permutated_names)
plt.ylabel('Mean recognition through all identities')
plt.title("Mean recognition through all identities depending of the model used")


def autolabel(rects, _stats):
    data_line, capline, barlinecols = rects.errorbar

    for err_segment, rect, s in zip(barlinecols[0].get_segments(), rects, _stats):
        error_bar_height_low = err_segment[0][1]  # Use height of error bar
        error_bar_height_sup = err_segment[1][1]  # Use height of error bar
        bar_height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, bar_height, f"{s.mean:.4f}", ha='center', va='bottom', bbox=dict(facecolor='lime', alpha=1))
        plt.text(rect.get_x() + rect.get_width() / 2.0, error_bar_height_low, f"{s.minmax[0]:.4f}", ha='center', va='bottom', bbox=dict(facecolor='yellow', alpha=1))
        plt.text(rect.get_x() + rect.get_width() / 2.0, error_bar_height_sup, f"{s.minmax[1]:.4f}", ha='center', va='bottom', bbox=dict(facecolor='yellow', alpha=1))


autolabel(bar, total_stats)

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

plt.tight_layout()
plt.savefig(f"images/total.png")

plt.close(fig)
