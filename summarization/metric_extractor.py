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

    drifted_df = df.iloc[np.r_[(90000/500):(110000/500), (190000/500):(210000/500), (290000/500):(310000/500)]]

    acc = df["accuracy"].mean()
    gmean = df["gmean"].mean()
    gmean_affected = df["class_{}".format(int(n_class)-1)].mean()
    
    acc_drifted = drifted_df["accuracy"].mean()
    gmean_drifted = drifted_df["gmean"].mean()
    gmean_affected_drifted = drifted_df["class_{}".format(int(n_class)-1)].mean()

    df = {
        "id": identifier,
        "classifier": classifier,
        "n_class": n_class,
        "acc": acc,
        "gmean": gmean,
        "gmean_affected": gmean_affected,
        "drift_acc": acc_drifted,
        "drift_gmean": gmean_drifted,
        "drift_gmean_affected": gmean_affected_drifted
    }

    results.append(df)

df_results = pd.DataFrame(results)
print (df_results)

accuracy = df_results.pivot_table("acc", ["id", "n_class"], "classifier")
gmean = df_results.pivot_table("gmean", ["id", "n_class"], "classifier")
gmean_affected = df_results.pivot_table("gmean_affected", ["id", "n_class"], "classifier")
accuracy.to_csv("summarized_acc.csv", )
gmean.to_csv("summarized_gmean.csv", )
gmean_affected.to_csv("summarized_gmean_affected.csv", )

accuracy_drifted = df_results.pivot_table("drift_acc", ["id", "n_class"], "classifier")
gmean_drifted = df_results.pivot_table("drift_gmean", ["id", "n_class"], "classifier")
gmean_affected_drifted = df_results.pivot_table("drift_gmean_affected", ["id", "n_class"], "classifier")
accuracy_drifted.to_csv("summarized_drifted_acc.csv", )
gmean_drifted.to_csv("summarized_drifted_gmean.csv", )
gmean_affected_drifted.to_csv("summarized_drifted_gmean_affected.csv", )