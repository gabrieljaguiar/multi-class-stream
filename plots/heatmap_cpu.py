import pandas as pd
import altair as alt

data = pd.read_csv("../summarization/cpu_comparison_plot_data.csv")

df_melt = pd.melt(data, id_vars=["Data Stream"])

print (df_melt)

base = alt.Chart(df_melt).encode(
    alt.X('variable:O', axis=alt.Axis(orient='top', labelAngle=-45, title="")),
    alt.Y('Data Stream:O', axis=alt.Axis(labels=False, ticks=False)),
)

heatmap = base.mark_rect().encode(
    alt.Color('value:Q', title="Relative time", scale=alt.Scale(scheme='redblue', reverse=True)),
        
).configure_legend(
    gradientLength=150,
    gradientThickness=15,
    titleFontSize=10,
    labelFontSize=12
)


(heatmap).properties(width=200, height=200).save("heatmap_cpu.pdf")
