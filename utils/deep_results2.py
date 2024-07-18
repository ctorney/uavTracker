import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

terrier = pd.read_csv("../data/alfs_terrier/results/real_tracker_comparison.csv")
terrier["detector"] = "poor"
squirrel = pd.read_csv("../data/alfs_squirrel/results/real_tracker_comparison.csv")
terrier["detector"] = "good"

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

# Plotting with seaborn
sns.set_theme(style="whitegrid")
g = sns.FacetGrid(
    linkresults, col="scenario", row="variant", hue="detector", height=4, aspect=1.5
)
