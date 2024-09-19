import altair as alt
import numpy as np
import pandas as pd
import os
from glob import glob

#Plotting accuracy over time for sudden drifts

classifiers = [
    #"HT",
    #"EFHT",
    #"SRP", 
    #"ARF", 
    "OneVsAll",
    "OneVsAll_Dummy",
    "OneVsAll_CIDDM"
]

number_of_classes = [5,10,15]
dfs = []

PATH = "../output"
for n in number_of_classes:
    for classifier in classifiers:
        out_file = "{}/{}_swap_cluster_global_sudden_{}.csv".format(PATH, classifier, n)
        df = pd.read_csv(out_file)
        df["classifier"] = classifier
        df["number_of_classes"] = n
        dfs.append(df)



df_full = pd.concat(dfs, axis=0)

metrics = ["accuracy", "gmean"]

for metric in metrics:
    plot = alt.Chart(df_full).mark_line().encode(
        x='idx',
        y='{}:Q'.format(metric),
        color="classifier:N",
        strokeWidth=alt.value(1),
        column="number_of_classes:N"
    )

    #chart = alt.vconcat()
    #for n in number_of_classes:
    #    chart |= plot.transform_filter(alt.datum.number_of_classes == n)

    plot.save("plot_{}.pdf".format(metric))

chart = alt.hconcat() 
for n in number_of_classes:
    df_partial = df_full[df_full["number_of_classes"] == n]
    plot = alt.Chart(df_partial).mark_line().encode(
        x='idx',
        y='class_{}:Q'.format(n-1),
        color="classifier:N",
        strokeWidth=alt.value(1),
        column="number_of_classes:N"
    )
    
    chart |= plot
chart.save("plot_gmean_drifted_class.pdf")