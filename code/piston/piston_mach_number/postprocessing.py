import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

sns.set_theme()

df = pd.read_csv("wiggles.csv", index_col=0)
df = df.drop(["alpha", "gamma"], axis=1)
print(df.head())
sns.pairplot(df, hue="wiggles")
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
