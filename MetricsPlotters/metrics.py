import os
import pandas as pd
import matplotlib.pyplot as plt


# Shared variables
label_fontsize = 16
tick_fontsize = 12
dpi = 200
save_dir = "pong"
gamma = 0.6
color_alpha = 0.3


# Plot 1
train_return_path = "/Users/erikpersson/PycharmProjects/AutoDrone/MetricsPlotters/plots/pong/total_return.csv"
eval_return_path = "/Users/erikpersson/PycharmProjects/AutoDrone/MetricsPlotters/plots/pong/total_return.csv"
fig_name = "total_return"
fig_title = ""
x_label = "Epoch"
y_label = "Total return"
human_baseline = 152

parent_dir = "plots/" + save_dir
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

df = pd.read_csv(train_return_path)
epochs_train = df.Step
tot_ret_train = df.Value

df = pd.read_csv(eval_return_path)
epochs_eval = df.Step
tot_ret_eval = df.Value

tot_ret_train_ema = tot_ret_train.ewm(alpha=1-gamma, adjust=True).mean()
tot_ret_eval_ema = (tot_ret_eval-20).ewm(alpha=1-gamma, adjust=True).mean()

fig = plt.figure()
ax = fig.subplots()
ax.plot(epochs_train, tot_ret_train, 'r-', label='_nolegend_', alpha=color_alpha)
ax.plot(epochs_train, tot_ret_train_ema, 'r-', label='Training')
ax.plot(epochs_eval, tot_ret_eval-20, 'g--', label='_nolegend', alpha=color_alpha)
ax.plot(epochs_eval, tot_ret_eval_ema, 'g--', label='Evaluation')
ax.plot([epochs_train.iloc[0], epochs_train.iloc[-1]], [human_baseline, human_baseline], 'b-.', label='Human baseline')
ax.set_xlabel(x_label, fontsize=label_fontsize)
ax.set_ylabel(y_label, fontsize=label_fontsize)
ax.set_title(fig_title)
ax.tick_params(axis='both', labelsize=tick_fontsize)
ax.legend(fontsize=tick_fontsize)
fig.savefig("plots/" + save_dir + "/" + fig_name, dpi=dpi)
plt.show()