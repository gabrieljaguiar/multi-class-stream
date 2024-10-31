import altair as alt
import numpy as np
import pandas as pd
import os
from glob import glob


def format_label(num):
    return "{}K".format(num // 1000)


onevall_models = [
    "OneVsAll-NC",
    "OneVsAll-DDM",
    "OneVsAll-GT",
    "OneVsAll-CIDDM",
]

tree_models = [
    "OneVsAll-GT",
    "OneVsAll-CIDDM",
    "HT",
    "DDM-HT",
    "EFHT",
]

ensemble_comparison = [
    "OneVsAll-CIDDM", "Bagging-CIDDM",
    "Bagging-GT",
    "SRP",
    "ARF",
    "ADWINBagging",
    "AdaBoost",
]

classifiers = ensemble_comparison


number_of_classes = [5, 10, 15]
localities = ["global", "local"]
speeds = ["gradual", "sudden"]
dfs = []

PATH = "../output/sycamore"
for n in number_of_classes:
    for speed in speeds:
        for locality in localities:
            for classifier in classifiers:
                out_file = "{}/{}_swap_cluster_{}_{}_{}.csv".format(
                    PATH, classifier, locality, speed, n
                )
                df = pd.read_csv(out_file, index_col=0)
                df["classifier"] = classifier
                df["locality"] = locality
                df["speed"] = speed
                df["number_of_classes"] = n
                dfs.append(df)
df_full = pd.concat(dfs, axis=0)

grouped = df_full.groupby(["locality", "speed", "number_of_classes"])
metrics = ["accuracy", "gmean", "kappa", "class"]
# alt.themes.enable('ggplot2')
for metric in metrics:
    for name, group in grouped:
        _, _, n_classes = name
        if "class" in metric:
            metric = "class_{}".format(n_classes-1)
            title = "Drifted class G-Mean"
        else:
            title = metric.title()
        plot = (
            alt.Chart(group, height=200, width=200)
            .mark_line(interpolate="monotone")
            .encode(
                x=alt.X("idx", scale=alt.Scale(domain=[0, 400*1000], padding=0))
                .axis(
                    offset=10,
                    tickCount=10,
                    labelExpr="datum.value % 100000 ? null : (datum.value / 1000) + 'k'",
                )
                .title("Instances"),
                y=alt.Y(metric, scale=alt.Scale(domain=[0, 1.025], padding=0.1))
                .axis(format="%", tickCount=10, offset=10)
                .title(title),
                color=alt.Color(
                    "classifier:N",
                    legend=alt.Legend(
                        columns=2,
                        orient="top",
                        title=None,
                        # legendY=-40,
                        direction="horizontal",
                        titleAnchor="middle",
                        labelFontSize=10,
                    ),
                ).scale(scheme="set2"),
                strokeWidth=alt.value(0.6),
            )
        )

        rules = (
            alt.Chart(
                pd.DataFrame(
                    {"drift": [100000, 200000, 300000], "color": ["red", "red", "red"]}
                )
            )
            .mark_rule(strokeDash=[4, 4, 4])
            .encode(x="drift", color=alt.value("red"), opacity=alt.value(0.4))
        )
        plot = plot + rules

        plot = plot.configure_view(
            stroke="black",
            strokeWidth=2,
            #clip=True
        )

        plot = plot.configure_axis(grid=True)

        file_config = "_".join(map(str, name))
        
        if "class" in metric:
            file_metric = "affected_gmean"
        else:
            file_metric = metric

        plot.save("ensemble/{}_{}.pdf".format(file_metric, file_config))
