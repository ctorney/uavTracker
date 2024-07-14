import os, math, yaml, argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from toy_data_generator import Mooveemodel
from utils import init_config

config_file = f"../experiments/alfs.yml"
with open(config_file, "r") as configfile:
    config = yaml.safe_load(configfile)
project_directory = config["project_directory"]

generator_config_file = f"../experiments/alfs_datasets.yml"
with open(generator_config_file, "r") as configfile:
    generator_config = yaml.safe_load(configfile)

d_speed = pd.DataFrame()
d_ang = pd.DataFrame()
mysettings = generator_config["settings"][2:]
nsettings = len(mysettings)
nline = 200
nsamples = 10000

for setting in mysettings:
    mu_s = generator_config[setting]["mu_s"]
    sigma_speed = generator_config[setting]["sigma_speed"]
    sigma_angular_velocity = generator_config[setting]["sigma_angular_velocity"]
    theta_speed = generator_config[setting]["theta_speed"]
    theta_angular_velocity = generator_config[setting]["theta_angular_velocity"]
    no_alfs = generator_config[setting]["no_alfs"]
    genmodel = generator_config[setting]["model"]

    mm = Mooveemodel(
        0,
        0,
        mu_s,
        sigma_speed,
        sigma_angular_velocity,
        theta_speed,
        theta_angular_velocity,
    )
    dmv = mm.prepMovement(nsamples=1000)
    d_speed["time"] = dmv["time"]
    d_speed[setting] = dmv["speed"]
    d_ang["time"] = dmv["time"]
    d_ang[setting] = dmv["angular_velocity"]

plt.clf()

# Create subplots: 4 columns (speed line, speed hist, angle line, angle hist) for each setting
fig, axs = plt.subplots(
    nsettings,  # One row per setting
    5,  # Four columns per setting
    figsize=(20, 5 * nsettings),  # Adjust width and height
    constrained_layout=True,
)

# Loop over the settings and plot
for iii, setting in enumerate(mysettings):
    axs[iii, 0].text(
        0.5,
        0.5,
        setting,
        fontsize=24,
        fontweight="bold",
        fontname="serif",
        ha="center",
        va="center",
    )
    axs[iii, 0].axis("off")  # Turn off the axis

    # Speed line plot
    sns.lineplot(data=d_speed[:nline], x="time", y=setting, ax=axs[iii, 1])
    axs[iii, 1].set_title(f"Line Plot of Speed for {setting}")

    # Speed histogram
    sns.histplot(d_speed[setting], bins=20, ax=axs[iii, 2])
    axs[iii, 2].set_title(f"Histogram of Speed for {setting}")

    # Angular velocity line plot
    sns.lineplot(data=d_ang[:nline], x="time", y=setting, ax=axs[iii, 3])
    axs[iii, 3].set_title(f"Line Plot of Angular Velocity for {setting}")

    # Angular velocity histogram
    sns.histplot(d_ang[setting], bins=20, ax=axs[iii, 4])
    axs[iii, 4].set_title(f"Histogram of Angular Velocity for {setting}")

# save as png
plt.savefig(f"{project_directory}/movement_plots.png")
