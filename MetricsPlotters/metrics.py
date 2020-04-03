import os
import pandas as pd
import matplotlib.pyplot as plt


# Edit these variables
save_dir = "pong"
metric_path = "/Users/erikpersson/PycharmProjects/AutoDrone/MetricsPlotters/plots/pong/total_return.csv"
fig_name = "total_return"
fig_title = ""
x_label = "Epoch"
y_label = "Total return"
label_fontsize = 16
tick_fontsize = 12
dpi = 200



parent_dir = "plots/" + save_dir
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

df = pd.read_csv(metric_path)
epochs = df.Step
metric = df.Value

fig = plt.figure()
ax = fig.subplots()
ax.plot(epochs, metric)
ax.set_xlabel(x_label, fontsize=label_fontsize)
ax.set_ylabel(y_label, fontsize=label_fontsize)
ax.set_title(fig_title)
ax.tick_params(axis='both', labelsize=tick_fontsize)
fig.savefig("plots/" + save_dir + "/" + fig_name, dpi=dpi)
