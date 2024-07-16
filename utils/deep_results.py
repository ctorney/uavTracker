import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("../data/alfs_terrier/results/tracker_comparison.csv")


df = a
# Assuming 'df' is already defined and contains your data
methods = ["torney", "sort", "beast"]
colors = ["blue", "green", "red"]
x = np.arange(len(df["sequence"]))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
for i, method in enumerate(methods):
    ax.bar(
        x + i * width,
        df[method],
        width,
        label=method,
        yerr=df[f"{method}_std"],
        capsize=5,
        color=colors[i],
    )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel("Sequence")
ax.set_ylabel("Performance")
ax.set_title("Performance of Methods by Sequence with Standard Deviation")
ax.set_xticks(x + width)
ax.set_xticklabels(df["sequence"])
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
