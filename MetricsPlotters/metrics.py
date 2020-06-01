import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Shared variables
label_fontsize = 16
tick_fontsize = 12
dpi = 300
save_dir = "exploration_comp"
gamma = 0.6
color_alpha = 0.3

if False:
    # Plot 1
    # change here
    metrics_path = "/Users/erikpersson/PycharmProjects/AutoDrone/MetricsPlotters/plots/exploration/Exploration_Eval_TotalReturn.csv"
    human_baseline = None
    fig_name = "exploration_prob_tot_return"
    metrics_label = ''
    fig_title = ""
    x_label = "Epoch"
    y_label = "Total return"
    ylim = [-10, 50]

    parent_dir = "plots/" + save_dir
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    df = pd.read_csv(metrics_path)
    epochs = df.Step
    metric = df.Value

    metric_ema = metric.ewm(alpha=1 - gamma, adjust=True).mean()

    fig = plt.figure()
    ax = fig.subplots()
    ax.set_ylim(ylim[0], ylim[1])
    ax.plot(epochs, metric, 'r-', label='_nolegend_', alpha=color_alpha)
    ax.plot(epochs, metric_ema, 'r-', label=metrics_label)
    if human_baseline is not None:
        ax.plot([epochs.iloc[0], epochs.iloc[-1]], [human_baseline, human_baseline], 'b-.', label='Human baseline')
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(fig_title)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    #ax.legend(fontsize=tick_fontsize)

    strFile = "plots/" + save_dir + "/" + fig_name
    fig.savefig(strFile, dpi=dpi, bbox_inches='tight')
    plt.show()

def fix_collisions(n_cells, max_steps):
    n_cells_tmp = np.ones((max_steps,)) * n_cells[-1]
    n_cells_tmp[0:len(n_cells)] = n_cells
    return n_cells_tmp

# Explored cells
if False:
    # Edit here
    max_step = 127
    data_dirs = ["epoch588", "random5", "random10"]
    fig_name = "explored_cells_comp"
    x_label = 'Steps'
    y_label = 'Explored Cells'
    fig_title = ''
    legend_labels = ["Neural Network Agent", "$\sigma = 5.0$", "$\sigma = 10.0$"]

    data_paths = []
    for dir in data_dirs:
        data_paths.append("plots/" + save_dir + "/" + dir)

    n_series = len(data_paths)
    fig = plt.figure()
    ax = fig.subplots()

    for i in range(n_series):
        n_files = 0
        avg_cells = np.zeros((127, ))
        for file in os.listdir(data_paths[i]):
            df = pd.read_csv(data_paths[i] + "/" + file)
            n_cells = np.array(df.Value)
            if len(n_cells) < max_step:
                n_cells = fix_collisions(n_cells, max_step)

            avg_cells += n_cells
            n_files += 1

        avg_cells /= n_files
        print(n_files)
        ax.plot(range(1,max_step+1), avg_cells, label=legend_labels[i])

    ax.legend(fontsize=tick_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(fig_title)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

# Objects
if True:
    # Edit here
    max_step = 127
    save_dir = "full/precision"
    data_dirs = ["epoch588","random5", "random10"]
    fig_name = "precision_new"
    x_label = 'Steps'
    y_label = 'Precision'
    fig_title = ''
    legend_labels = ["Neural Network Agent", "$\sigma = 5.0$", "$\sigma = 10.0$"]

    data_paths = []
    for dir in data_dirs:
        data_paths.append("plots/" + save_dir + "/" + dir)

    n_series = len(data_paths)
    fig = plt.figure()
    ax = fig.subplots()

    for i in range(n_series):
        n_files = 0
        avg_cells = np.zeros((127, ))
        for file in os.listdir(data_paths[i]):
            df = pd.read_csv(data_paths[i] + "/" + file)
            n_cells = np.array(df.Value)
            if len(n_cells) < max_step:
                n_cells = fix_collisions(n_cells, max_step)

            avg_cells += n_cells
            n_files += 1

        avg_cells /= n_files
        print(n_files)
        ax.plot(range(1,max_step+1), avg_cells, label=legend_labels[i])

    ax.legend(fontsize=tick_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(fig_title)
    ax.tick_params(axis='both', labelsize=tick_fontsize)


strFile = "plots/" + save_dir + "/" + fig_name
fig.savefig(strFile, dpi=dpi, bbox_inches='tight')
plt.show()