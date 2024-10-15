import altair as alt
import numpy as np
import pandas as pd
import os
from glob import glob

models = [
    #"ADWINBagging",
    "ARF",
    #"AdaBoost",
    #"DDM-HT",
    #"EFHT",
    "HT",
    #"OneVsAll-CIDDM",
    #"OneVsAll-DDM",
    "OneVsAll-GT",
    "OneVsAll-NC",
    #"SRP",
]


streams = [
    #"semi_synth_1_to_3_gradual",
    #"semi_synth_1_to_3_sudden",
    #"semi_synth_1_to_6_gradual",
    #"semi_synth_1_to_6_sudden",
    #"semi_synth_6_to_3_gradual",
    #"semi_synth_6_to_3_sudden",
    "semi_synth_6_to_3_sudden_global_change",
]

PATH = "../output/semi-synth/"
for stream in streams:
    dfs = []
    for model in models:
        out_file = "{}/{}_{}.csv".format(PATH, model, stream)
        df = pd.read_csv(out_file)
        df["classifier"] = model
        dfs.append(df)

    df_full = pd.concat(dfs, axis=0)

    plot = (
        alt.Chart(df_full)
        .mark_line()
        .encode(
            x="idx",
            y="accuracy",
            color="classifier:N",
            strokeWidth=alt.value(1),
        )
    )

    plot.save("{}.pdf".format(stream))

    plot = (
        alt.Chart(df_full)
        .mark_line()
        .encode(
            x="idx",
            y="class_5",
            color="classifier:N",
            strokeWidth=alt.value(1),
        )
    )

    plot.save("g_mean_{}.pdf".format(stream))