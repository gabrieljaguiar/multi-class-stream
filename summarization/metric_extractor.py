import os 
from glob import glob
import pandas as pd
import numpy as np

models = [
    "id", "HT", "EFHT", "SRP", "ARF", "OneVsAll", "OneVsAll_CIDDM", "OneVsAll_Dummy"
]


output_files = glob("../output/*.csv")

results = []

for output_file in output_files:
    classifier = output_file.split("_")[0].split("/")[-1]
    if classifier == "OneVsAll":
        classifier = "_".join(output_file.split("_")[0:2]).split("/")[-1]
    n_class = output_file.split("_")[-1].split(".")[0]
    identifier = "_".join(output_file.split("_")[-5:-1])
    
    df = pd.read_csv(output_file)

    drifted_df = df.iloc[np.r_[(100000/500):(150000/500), (200000/500):(250000/500), (300000/500):(350000/500)]]

    acc = df["accuracy"].mean()
    gmean = df["gmean"].mean()
    kappa = df["kappa"].mean()
    gmean_affected = df["class_{}".format(int(n_class)-1)].mean()
    mem_used = df["mem_usage"].mean()
    cpu_time = df["cpu_time"].tail(1).values[0]
    
    acc_drifted = drifted_df["accuracy"].mean()
    gmean_drifted = drifted_df["gmean"].mean()
    kappa_drifted = drifted_df["kappa"].mean()
    gmean_affected_drifted = drifted_df["class_{}".format(int(n_class)-1)].mean()


    df = {
        "id": identifier,
        "classifier": classifier,
        "n_class": n_class,
        "acc": acc,
        "gmean": gmean,
        "kappa": kappa,
        "mem_usage":mem_used,
        "cpu_time": cpu_time,
        "gmean_affected": gmean_affected,
        "drift_acc": acc_drifted,
        "drift_gmean": gmean_drifted,
        "drift_kappa": gmean_drifted,
        "drift_gmean_affected": gmean_affected_drifted
    }

    results.append(df)

df_results = pd.DataFrame(results)
print (df_results)

accuracy = df_results.pivot_table("acc", ["id", "n_class"], "classifier")
gmean = df_results.pivot_table("gmean", ["id", "n_class"], "classifier")
kappa = df_results.pivot_table("kappa", ["id", "n_class"], "classifier")
gmean_affected = df_results.pivot_table("gmean_affected", ["id", "n_class"], "classifier")
mem_usage_df = df_results.pivot_table("mem_usage", ["id", "n_class"], "classifier")
cpu_time_df = df_results.pivot_table("cpu_time", ["id", "n_class"], "classifier")
cpu_time_df = cpu_time_df.apply(lambda x: x/x.min(), axis=1)
accuracy.to_csv("summarized_acc.csv", )
kappa.to_csv("summarized_kappa.csv", )
gmean.to_csv("summarized_gmean.csv", )
gmean_affected.to_csv("summarized_gmean_affected.csv", )
mem_usage_df.to_csv("summarized_mem_usage.csv")
cpu_time_df.to_csv("summarized_cpu_time.csv")

accuracy_drifted = df_results.pivot_table("drift_acc", ["id", "n_class"], "classifier")
gmean_drifted = df_results.pivot_table("drift_gmean", ["id", "n_class"], "classifier")
kappa_drifted = df_results.pivot_table("drift_kappa", ["id", "n_class"], "classifier")
gmean_affected_drifted = df_results.pivot_table("drift_gmean_affected", ["id", "n_class"], "classifier")
accuracy_drifted.to_csv("summarized_drifted_acc.csv", )
gmean_drifted.to_csv("summarized_drifted_gmean.csv", )
kappa_drifted.to_csv("summarized_drifted_kappa.csv", )
gmean_affected_drifted.to_csv("summarized_drifted_gmean_affected.csv", )