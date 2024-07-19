import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

terrier = pd.read_csv("../data/alfs_terrier/results/tracker_results_samples.csv")
terrier["detector"] = "poor"
squirrel = pd.read_csv("../data/alfs_squirrel/results/tracker_results_samples.csv")
squirrel["detector"] = "good"

linkresults = pd.concat([terrier, squirrel])

sequence_lookup = {
    "solitary.txt": (1, "diff", "Z"),
    "group.txt": (3, "diff", "Z"),
    "pack.txt": (10, "diff", "Z"),
    "herd.txt": (20, "diff", "Z"),
    "group_identical.txt": (3, "same", "Z"),
    "pack_identical.txt": (10, "same", "Z"),
    "herd_identical.txt": (20, "same", "Z"),
    "A_test_solitary.txt": (1, "diff", "A"),
    "A_test_group.txt": (3, "diff", "A"),
    "A_test_pack.txt": (10, "diff", "A"),
    "A_test_herd.txt": (20, "diff", "A"),
    "B_test_solitary.txt": (1, "diff", "B"),
    "B_test_group.txt": (3, "diff", "B"),
    "B_test_pack.txt": (10, "diff", "B"),
    "B_test_herd.txt": (20, "diff", "B"),
}
linkresults["n_objects"] = linkresults["sequence"].map(lambda x: sequence_lookup[x][0])
linkresults["variant"] = linkresults["sequence"].map(lambda x: sequence_lookup[x][1])
linkresults["scenario"] = linkresults["sequence"].map(lambda x: sequence_lookup[x][2])

##############
############
#############
#########
######
################
####
####################
######### compare diff and same on 'good'
##################
#################
########
#####
# Load your DataFrame
# linkresults = pd.read_csv('your_data.csv')  # Assuming data is in CSV for example

# Filter data
# filtered_data = linkresults[(linkresults["scenario"] == "Z")]
filtered_data = linkresults[
    (linkresults["scenario"] == "Z") & (linkresults["detector"] == "good")
]

# Melt the DataFrame to get 'torney', 'sort', 'beast' into a single column
melted_data = filtered_data.melt(
    id_vars=["detector", "n_objects", "variant", "scenario"],
    value_vars=["torney", "sort", "beast"],
    var_name="method",
    value_name="performance",
)
melted_data["detector_variant"] = melted_data[["detector", "variant"]].apply(
    lambda x: f"{x[0]}_{x[1]}", axis=1
)
# Plot using seaborn's catplot
g = sns.catplot(
    data=melted_data,
    x="method",
    y="performance",
    hue="variant",
    col="n_objects",
    kind="box",
    ci="sd",  # standard deviation for the error bars
    aspect=0.6,
)


# Adjust legend and axis labels
g.set_axis_labels("Method", "Performance")
g.set_titles("Objects: {col_name}")
g.add_legend(title="Variant")

# Save the figure
plt.savefig("plots/results_variant.png")

plt.show()

##############
############
#############
#########
######
################
####
####################
######### compare good and bad!
##################
#################
########
#####
filtered_data = linkresults[
    (linkresults["scenario"] == "Z") & (linkresults["variant"] == "diff")
]

# Melt the DataFrame to get 'torney', 'sort', 'beast' into a single column
melted_data = filtered_data.melt(
    id_vars=["detector", "n_objects", "variant", "scenario"],
    value_vars=["torney", "sort", "beast"],
    var_name="method",
    value_name="performance",
)
melted_data = melted_data[(melted_data["method"] == "beast")]

# Plot using seaborn's catplot
g = sns.catplot(
    data=melted_data,
    x="n_objects",
    y="performance",
    hue="detector",
    kind="box",
    ci="sd",  # standard deviation for the error bars
    aspect=0.6,
)


# Adjust legend and axis labels
g.set_axis_labels("Number of Objects", "Average IoU")
g.add_legend(title="Detector quality on scenario Z")

# Save the figure
plt.savefig("plots/results_dets.png")

plt.show()
##############
############
#############
#########
######
################
####
####################
#########
##################
#################
########
#####


# Compute mean performance grouped by 'method', 'variant', and 'n_objects'
mean_results = (
    melted_data.groupby(["method", "variant", "n_objects"])["performance"]
    .agg(["mean", "std"])
    .reset_index()
)

# Rename the columns for clarity
mean_results.rename(
    columns={"mean": "Mean Performance", "std": "Standard Deviation"}, inplace=True
)

# Display or save the DataFrame
print(mean_results)
# Optionally save to CSV
# mean_results.to_csv("plots/mean_performance_results.csv", index=False)
# print tex
print(mean_results.to_latex(index=False))

##############
############
#############
#########
######
################
####
####################
#########
##################
#################
########
#####

filtered_data = linkresults[
    (linkresults["detector"] == "good") & (linkresults["variant"] == "diff")
]
# Melt the DataFrame to get 'torney', 'sort', 'beast' into a single column
melted_data = filtered_data.melt(
    id_vars=["detector", "n_objects", "variant", "scenario"],
    value_vars=["torney", "sort", "beast"],
    var_name="method",
    value_name="performance",
)
melted_data["n_objects"] = melted_data["n_objects"].astype(str)
#
g = sns.catplot(
    data=melted_data,
    x="method",
    y="performance",
    hue="n_objects",
    col="scenario",
    col_order=["A", "B", "Z"],  # Specify the order of the columns
    kind="box",
    ci="sd",  # Standard deviation for the error bars
    aspect=0.6,
)

# Adjust the axis labels and plot title
g.set_axis_labels("Variant", "Performance")
g.set_titles("Objects: {col_name}")
plt.savefig(f"plots/results_main.png")
plt.show()

#
##############
############
#############
#########
######
################
####
####################
#########
##################
#################
########
#####
