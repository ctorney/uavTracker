import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("../data/alfs_terrier/results/tracker_comparison.csv")
df2 = pd.read_csv("../data/alfs_terrier_identical/results/tracker_comparison.csv")

# Assume df1 and df2 are defined DataFrames similar to your initial structure
methods = ["torney", "sort", "beast"]
colors = ["blue", "green", "red"]
width = 0.25  # the width of the bars

# Setup figure and axes
fig, axes = plt.subplots(2, 1, figsize=(14, 12))  # 2 rows, 1 column


# Function to plot each DataFrame
def plot_df(ax, df, title):
    x = np.arange(len(df["sequence"]))  # the label locations
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

    ax.set_xlabel("Sequence")
    ax.set_ylabel("Performance")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["sequence"])
    ax.legend()


# Plot df1 and df2
plot_df(
    axes[0], df1, "Performance of Methods by Sequence with Standard Deviation (df1)"
)
plot_df(
    axes[1], df2, "Performance of Methods by Sequence with Standard Deviation (df2)"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

resfile = "../data/alfs_terrier/results/results_file.yml"
with open(resfile, "r") as f:
    res = yaml.safe_load(f)

res_squirrel_train = pd.Series(res["AP"]["all_sets"]["terrier"]["phase_two"]).apply(
    lambda x: round(100 * x, 2)
)
res_squirrel_test = pd.Series(res["AP"]["test"]["terrier"]["phase_two"]).apply(
    lambda x: round(100 * x, 2)
)
res_terrier_train = pd.Series(res["AP"]["all_sets"]["terrier"]["phase_one"]).apply(
    lambda x: round(100 * x, 2)
)
res_terrier_test = pd.Series(res["AP"]["test"]["terrier"]["phase_one"]).apply(
    lambda x: round(100 * x, 2)
)

# Create a table of results for squirrel and terrier with subheading train/test with each row being a value with index the same as the series
det_results = pd.DataFrame(
    {
        "Squirrel-train": res_squirrel_train,
        "Squirrel-test": res_squirrel_test,
        "Terrier-train": res_terrier_train,
        "Terrier-test": res_terrier_test,
    }
)
# change name of the index column to mAP threshold
det_results.index.name = "mAP threshold"

# print det_results to latex, one decimal point
print(det_results.to_latex(index=True, float_format="%.1f"))
