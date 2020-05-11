import os
import pandas as pd
import matplotlib.pyplot as plt


# Shared variables
label_fontsize = 16
tick_fontsize = 12
dpi = 300
save_dir = "pong"
gamma = 0.6
color_alpha = 0.3

if True:
    # Plot 1
    #change here
    metrics_path = "/Users/erikpersson/PycharmProjects/AutoDrone/MetricsPlotters/plots/pong/Pong-Eval_TotalReturn.csv"
    human_baseline =None
    fig_name = "pong_eval_tot_return"
    metrics_label = 'Evaluation'
    fig_title = ""
    x_label = "Epoch"
    y_label = "Total return"
    ylim = [-22, 22]

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
    ax.legend(fontsize=tick_fontsize)

    strFile = "plots/" + save_dir + "/" + fig_name
    fig.savefig(strFile, dpi=dpi, bbox_inches='tight')
    plt.show()
