import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

sns.set_theme()

df = pd.read_csv("wiggles.csv", index_col=0)
df = df.drop(["alpha", "gamma"], axis=1)

rename = dict(
    a0="$a_0$",
    delta="$\\delta$",
    omega="$\\omega$",
    forcing="$\\frac{\\delta L_0 \\omega}{a_0}$",
    eta="$\\eta$",
    wiggles="Wiggles",
)
df = df.rename(columns=rename)
sns.pairplot(df, hue="Wiggles")
plt.show()

# df = df.round(2)

# cols = list(df.columns)
# cols.remove("wiggles")
# fig = px.scatter_matrix(
#     df,
#     dimensions=cols,
#     color="wiggles",
#     hover_data=df.columns,
# )
# fig.show()
