import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('log_files_with_accuracy.csv')
df.loc[df['perturb'].isna(), 'perturb'] = 'No Perturbation'

# Capitalize
df['perturb'] = df['perturb'].apply(
    lambda x: " ".join(token.capitalize() for token in x.split(" ")))

df.loc[df['prompt'] == '0cot', 'prompt'] = "0CoT"
df.loc[df['prompt'] == 'cot', 'prompt'] = "CoT"
df.loc[df['prompt'] == 'ltm', 'prompt'] = "LTM"

df = df.rename(columns={'prompt': 'Prompt',
                        'perturb': 'Perturbation',
                        'propmt': 'Prompt',
                        'accuracy': 'Accuracy',
                        'shots': 'Shots', })

df["Combination"] = df.apply(
    lambda x: x["Prompt"] + ((f", {x['perturb_exemplar']} Perturbed Exemplar") if x["Prompt"] != "0CoT" else ""), axis=1
)


df2 = df[(df['Shots']) == 1 & ((df['Perturbation'] == 'No Perturbation')
         | (df['perturb_exemplar'] == 0))]

# create error bar representing binomial proportion standard error
p = df2["Accuracy"]

df3 = df2.copy()
df4 = df2.copy()

df3["Accuracy"] = p + 1.96 * np.sqrt(p * (1 - p) / 8791)
df4["Accuracy"] = p - 1.96 * np.sqrt(p * (1 - p) / 8791)

df2 = pd.concat([df2, df3, df4])

sns.set_theme(style="darkgrid")

# plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=df2, kind="bar",
    x="Prompt", y="Accuracy", hue="Perturbation",
    palette="bright", alpha=.6, height=2.5, aspect=1.8,
    legend=False, errorbar=("pi", 100), errwidth=2,
    hue_order=['No Perturbation', 'Typo', 'Synonym', 'Repetition', 'Shortcut']
)

# turn off x labels
g.set_xlabels("")

plt.ylim(0.5, 0.8)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('images/simple_perturb.png', bbox_inches='tight')

# df2 = df[((df['Shots'] == 1) | (df["Prompt"] == "0CoT"))
#          & (df['Perturbation'] != 'No Perturbation')]
# df2 = df2[(df2["Prompt"] != "0CoT") | (df2["perturb_exemplar"] == 0)]

# plt.figure(figsize=(10, 4))
# g = sns.catplot(
#     data=df2, kind="bar",
#     x="Perturbation", y="Accuracy", hue="Combination",
#     palette="dark", alpha=.6, height=4, aspect=2,
# )
# plt.ylim(0.5, 0.85)

# df2 = df2.sort_values(['Perturbation', 'Prompt'])
# df2["Accuracy"] = df2["Accuracy"] - \
#     df.groupby('Perturbation')["Accuracy"].transform('first')

# plt.savefig('images/perturb_exemplar.png', bbox_inches='tight')

df2 = df[(df['Shots'] == 8) & (df['Perturbation'] != 'No Perturbation')]

# create error bar representing binomial proportion standard error
p = df2["Accuracy"]

df3 = df2.copy()
df4 = df2.copy()

df3["Accuracy"] = p + 1.96 * np.sqrt(p * (1 - p) / 8791)
df4["Accuracy"] = p - 1.96 * np.sqrt(p * (1 - p) / 8791)

df2 = pd.concat([df2, df3, df4])

g = sns.catplot(
    data=df2, x="Prompt", y="Accuracy", col="Perturbation",
    hue="perturb_exemplar",
    kind="bar", height=3, aspect=1, palette="crest", alpha=.6,
    col_order=['Typo', 'Synonym', 'Repetition', 'Shortcut'],
    errorbar=("pi", 100), errwidth=2,
    legend=False
)
g.set_titles("{col_name}")
plt.ylim(0.55, 0.85)

# turn off x labels
g.set_xlabels("")

# get figure of g
# g.fig.supxlabel("Prompt")

df2 = df[(df["Prompt"] == "0CoT") & (df["perturb_exemplar"] == 0)]

# get axes of g
axes = g.axes.flatten()
axes[0].axhline(df2.iloc[3, 7], ls='--', color='red')
axes[1].axhline(df2.iloc[2, 7], ls='--', color='red')
axes[2].axhline(df2.iloc[0, 7], ls='--', color='red')
axes[3].axhline(df2.iloc[1, 7], ls='--', color='red')

# legend right side of plot
plt.legend(loc='upper center', bbox_to_anchor=(1.4, 0.9),
           title="# Perturbed\nExemplars")
# plt.legend(loc='upper center', bbox_to_anchor=(-1.2, -0.32),
#            ncol=5, title="Number of Perturbed Exemplars out of 8")

plt.savefig('images/perturb_shots.png', bbox_inches='tight')
