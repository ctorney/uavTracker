import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def print_tex_table(ldf, methods):
    ldf = linkresults
    for method in methods:
        m_std = f"{method}_std"
        ldf[method] = ldf.apply(
            lambda x: f"{round(100*x[method])}({round(100*x[m_std])})", axis=1
        )
    backslash = "\\"

    ldf.columns = [
        (x.replace("_", f"{backslash}textunderscore ") if x.find("_") != -1 else x)
        for x in ldf.columns
    ]

    latex_table = ldf.to_latex(
        float_format="%.2f", index_names=False, index=False, bold_rows=True
    )
    # Manual adjustments to the header
    header_line_index = (
        latex_table.find("\\toprule") + 8
    )  # Finding the position after \toprule
    header_line_end_index = latex_table.find(
        "\\midrule"
    )  # Finding the position of \midrule
    header_line = latex_table[header_line_index:header_line_end_index].strip()

    # Replace the auto-generated header with the custom header
    custom_header = " & ".join(ldf.columns) + " \\\\\n\\midrule"
    latex_table = latex_table.replace(header_line, custom_header)

    centered_latex_table = (
        "\\begin{table}[ht]\n\\centering\n" + latex_table + "\\end{table}"
    )
    print(centered_latex_table)
    return centered_latex_table


terrier = pd.read_csv("../data/alfs_terrier/results/real_tracker_comparison.csv")
squirrel = pd.read_csv("../data/alfs_squirrel/results/tracker_comparison.csv")

n_objects = [1, 3, 10, 20, 3, 10, 20, 1, 3, 10, 20, 1, 3, 10, 20]
variant = ["distinguished"] * 4 + ["identical"] * 3 + ["distinguished"] * 8
scenario = ["Z"] * 7 + ["A"] * 4 + ["B"] * 4

terrier["n_objects"] = n_objects
terrier["variant"] = variant
terrier["scenario"] = scenario
terrier["tracker"] = "terrier"

squirrel["n_objects"] = n_objects
squirrel["variant"] = variant
squirrel["scenario"] = scenario
squirrel["tracker"] = "squirrel"

linkresults = pd.concat([terrier, squirrel])


# Assume df1 and df2 are defined DataFrames similar to your initial structure
methods = ["sort", "torney", "beast"]
mandm = ["sort", "sort_std", "torney", "torney_std", "beast", "beast_std"]
colors = ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", "#a6d854"]
width = 0.25  # the width of the bars
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_results(linkresults, methods, mandm, width, colors):
    width = 0.15  # the width of the bars
    # Get unique n_objects
    unique_n_objects = linkresults["n_objects"].unique()

    # Setup the overall figure for subplots
    fig, axes = plt.subplots(len(unique_n_objects), 1, figsize=(10, 4 * len(unique_n_objects)))  # Adjust height based on number of subplots

    if len(unique_n_objects) == 1:  # If only one subplot, axes is not a list
        axes = [axes]

    for idx, no in enumerate(unique_n_objects):
        # Apply filter to data
        the_mask = (linkresults["tracker"] == "terrier") & (linkresults["n_objects"] == no) & (linkresults["variant"] == "distinguished")
        ln = linkresults[the_mask]
        fw_cols = mandm + ["scenario"]
        df = ln[fw_cols]

        # Get the specific axis for plotting
        ax = axes[idx]

        # Generate plot for the current dataframe slice
        x = np.arange(len(df["scenario"]))  # the label locations
        for i, method in enumerate(methods):
            bars = ax.bar(
                x + i * width,
                df[method],
                width,
                label=method,
                yerr=df[f"{method}_std"],
                capsize=5,
                color=colors[i]
            )
            for bar in bars:
                yval = bar.get_height()
                dispval = str(int(100*yval))
                ax.text(bar.get_x() + (bar.get_width()  / 2) +0.05, yval + 0.05, dispval, ha='center', va='bottom', fontsize=12, rotation=0)


        # Set axis labels and title
        ax.set_xlabel("Sequence")
        ax.set_ylabel("Average IoU")
        ax.set_title(f"Performance of Methods by Sequence with Standard Deviation {no}")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(df["scenario"])
        ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
# Assume linkresults, methods, mandm, width, colors are defined
# plot_combined_results(linkresults, methods, mandm, 0.35, colors)

#just for useless identity plot
def plot_df2(df, title):
    # Setup the plot
    met_nobj = [
        "beast_1", "beast_3", "beast_10", "beast_20", ,
        "sort_1", "sort_3", "sort_10", "sort_20",
        "torney_1", "torney_3", "torney_10", "torney_20",
    ]

    # Define colors for each variant explicitly
    variant_colors = {
        'distinguished': '#66c2a5',  # Greenish
        'identical': '#fc8d62'       # Orangish
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.35  # Bar width
    x = np.arange(len(met_nobj))  # x locations for the groups
    added_legend = []

    # Plot bars for each specified method and object count
    for i, col in enumerate(met_nobj):
        std_col = col + "_std"  # Standard deviation column
        bar_positions = x[i] + np.arange(len(df["variant"].unique())) * width

        for k, variant in enumerate(df["variant"].unique()):
            mean_values = df.loc[df["variant"] == variant, col].values
            std_values = df.loc[df["variant"] == variant, std_col].values

            # Decide the label: add to legend only if not already added
            label = variant if variant not in added_legend else ""
            if label:
                added_legend.append(variant)

            # Plot the bars
            ax.bar(
                bar_positions[k],  # Adjust position for each variant
                mean_values,
                width,
                label=label,
                yerr=std_values,
                capsize=5,
                alpha=0.75,
                color=variant_colors[variant]  # Use the specified color for the variant
            )

    # Set plot details
    ax.set_xlabel("Methods and Object Counts")
    ax.set_ylabel("Values")
    ax.set_title(title)
    # Adjust tick positions to be centered between the bars of two variants
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(met_nobj, rotation=45)
    ax.legend(title="Variant")

    plt.tight_layout()
    plt.show()

# Function to plot each DataFrame
def plot_df(df, title, xvar):
    # Setup single figure as ax
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # 2 rows, 1 colu

    x = np.arange(len(df[xvar]))  # the label locations
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
    ax.set_ylabel("Average IoU")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df[xvar])
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


full_variant_columns = ["variant", "n_objects"]
the_mask = (linkresults["tracker"] == "terrier") & (linkresults["scenario"] == "Z")
ln = linkresults[the_mask]
ln["full_variant"] = ln.apply(
    lambda x: f"{'_'.join([str(x[col]) for col in full_variant_columns])}", axis=1
)
fw_cols = mandm + ["full_variant"]
a = ln[fw_cols]



####################
###### group size
##########################
#sort linkresult by scenario and secondarily by n_object
linkresults = linkresults.sort_values(by=["scenario", "n_objects"])
plot_combined_results(linkresults, methods, mandm, width, colors)

for no in linkresults["n_objects"].unique():
    the_mask = (linkresults["tracker"] == "terrier") & (linkresults["n_objects"] == no) & (linkresults["variant"] == "distinguished")
    ln = linkresults[the_mask]
    fw_cols = mandm + ["scenario"]
    a = ln[fw_cols]
    print(a)
    plot_df(a, f"Performance of Methods by Sequence with Standard Deviation (df1) {no}", "scenario")


#########################
# identical vs distingusihed
############################
the_mask = (linkresults["tracker"] == "terrier") & (linkresults["scenario"] == "Z")
ln = linkresults[the_mask]
fw_cols = mandm + ["variant", "n_objects"]
a = ln[fw_cols]
new_cols = [
    f"{col}_{std}_{n}"
    for n in a["n_objects"].unique()
    for col in methods
    for std in ["", "std"]
]

# Pivot the DataFrame.
a_pivot = a.melt(
    id_vars=["variant", "n_objects"],
    value_vars=[col for pair in [(c, f"{c}_std") for c in methods] for col in pair],
)
a_pivot["variable"] = a_pivot["variable"] + "_" + a_pivot["n_objects"].astype(str)
a_pivot = a_pivot.pivot(
    index="variant", columns="variable", values="value"
).reset_index()
# Renaming columns as specified
new_columns = {}
for col in a_pivot.columns:
    parts = col.split("_")
    if "std" in parts:
        new_col = "_".join(
            [parts[0], parts[2], parts[1]]
        )  # Change order to method_number_std
        new_columns[col] = new_col
a_pivot.rename(columns=new_columns, inplace=True)


# Assume a_pivot is defined as shown previously
plot_df2(a_pivot, "Comparison of Methods by Variant and Object Count")

#########################

############################
plot_df(
    a,
    "Performance of Methods by Sequence with Standard Deviation (df1)",
)
print_tex_table(a, methods)
plot_df(
    linkresults[
        (linkresults["tracker"] == "terrier") & (linkresults["n_objects"] == 1)
    ],
    "Performance of Methods by Sequence with Standard Deviation (df1)",
)
plot_df(
    linkresults[
        (linkresults["tracker"] == "terrier") & (linkresults["scenario"] == "Z")
    ],
    "Performance of Methods by Sequence with Standard Deviation (df1)",
)
#  plot_df(
#     axes, linkresults[linkresults['tracker']=='terrier'], "Performance of Methods by Sequence with Standard Deviation (df1)"
# )


########################
####### Detectors ######
########################


resfile_ter = "../data/alfs_terrier/results/results_file.yml"
resfile_sqr = "../data/alfs_squirrel/results/results_file.yml"
with open(resfile_ter, "r") as f:
    res_ter = yaml.safe_load(f)
with open(resfile_sqr, "r") as f:
    res_sqr = yaml.safe_load(f)

res_squirrel_train = pd.Series(res_sqr["AP"]["all_sets"]["squirrel"]["phase_two"]).apply(
    lambda x: round(100 * x, 2)
)
res_squirrel_test = pd.Series(res_sqr["AP"]["test"]["squirrel"]["phase_two"]).apply(
    lambda x: round(100 * x, 2)
)
res_terrier_train = pd.Series(res_ter["AP"]["all_sets"]["terrier"]["phase_one"]).apply(
    lambda x: round(100 * x, 2)
)
res_terrier_test = pd.Series(res_ter["AP"]["test"]["terrier"]["phase_one"]).apply(
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
