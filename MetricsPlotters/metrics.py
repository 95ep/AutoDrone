import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

crashes_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/Downloaded_metrics/basic23-invalid-trgts-continued/Eval_nDones.csv"
terminations_correct_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/Downloaded_metrics/basic23-invalid-trgts-continued/Eval_nTerminationsCorrect.csv"
terminations_incorrect_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/Downloaded_metrics/basic23-invalid-trgts-continued/Eval_nTerminationsIncorrect.csv"
total_return_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/Downloaded_metrics/basic23-invalid-trgts-continued/Eval_TotalReturn.csv"

metric_labels = ["# Crashes", "# Terminations Correct", "# Terminations Incorrect", "Total return"]
metrics = []
df = pd.read_csv(crashes_path)
epochs = df.Step
metrics.append(df.Value)

df = pd.read_csv(terminations_correct_path)
metrics.append(df.Value)

df = pd.read_csv(terminations_incorrect_path)
metrics.append(df.Value)

df = pd.read_csv(total_return_path)
metrics.append(df.Value)



for i, metric in enumerate(metrics):
    label = metric_labels[i]
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(epochs, metric)
    ax.set_xlabel("Epoch")
    ax.set_title(label)
    fig.savefig(label + ".png")
