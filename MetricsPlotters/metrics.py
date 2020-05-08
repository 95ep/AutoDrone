import os
import pandas as pd
import matplotlib.pyplot as plt


# Shared variables
label_fontsize = 16
tick_fontsize = 12
dpi = 300
save_dir = "local_nav"
gamma = 0.6
color_alpha = 0.3

if True:
    # Plot 1
    #change here
    metrics_path = "D:/Exjobb2020ErikFilip/AutoDrone/MetricsPlotters/plots/local_nav/nCrashes.csv"
    human_baseline =2
    fig_name = "n_collisions"
    fig_title = ""
    x_label = "Epoch"
    y_label = "# collisions"
    ylim = [0, 50]

    parent_dir = "plots/" + save_dir
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    df = pd.read_csv(metrics_path)
    epochs_train = df.Step
    tot_ret_train = df.Value

    tot_ret_train_ema = tot_ret_train.ewm(alpha=1-gamma, adjust=True).mean()

    fig = plt.figure()
    ax = fig.subplots()
    ax.set_ylim(ylim[0], ylim[1])
    ax.plot(epochs_train, tot_ret_train, 'r-', label='_nolegend_', alpha=color_alpha)
    ax.plot(epochs_train, tot_ret_train_ema, 'r-', label='Training')
    ax.plot([epochs_train.iloc[0], epochs_train.iloc[-1]], [human_baseline, human_baseline], 'b-.', label='Human baseline')
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(fig_title)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=tick_fontsize)

    strFile = "plots/" + save_dir + "/" + fig_name
    fig.savefig(strFile, dpi=dpi, bbox_inches='tight')
    plt.show()
