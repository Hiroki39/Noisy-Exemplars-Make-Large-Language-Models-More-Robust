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
    lambda x: x["Prompt"] + ((", Perturbed Exemplar" if x["perturb_exemplar"] else ", Unperturbed Exemplar") if x["Prompt"] != "0CoT" else ""), axis=1
)


df2 = df[(df['Shots']) == 1 & ((df['Perturbation'] == 'No Perturbation')
         | (~df['perturb_exemplar']))]

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=df2, kind="bar",
    x="Prompt", y="Accuracy", hue="Perturbation",
    palette="dark", alpha=.6, height=4, aspect=2,
    hue_order=['No Perturbation', 'Typo', 'Repetition', 'Synonym', 'Shortcut'],
)

plt.savefig('images/simple_perturb.png', bbox_inches='tight')

df2 = df[((df['Shots'] == 1) | (df["Prompt"] == "0CoT"))
         & (df['Perturbation'] != 'No Perturbation')]
df2 = df2[(df2["Prompt"] != "0CoT") | (~df2["perturb_exemplar"])]

plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=df2, kind="bar",
    x="Perturbation", y="Accuracy", hue="Combination",
    palette="dark", alpha=.6, height=4, aspect=2,
)

df2 = df2.sort_values(['Perturbation', 'Prompt'])
df2["Accuracy"] = df2["Accuracy"] - \
    df.groupby('Perturbation')["Accuracy"].transform('first')

plt.savefig('images/perturb_exemplar.png', bbox_inches='tight')

df2 = df[df['perturb_exemplar'] & (df['Prompt'] != "0CoT") & (
    df['Perturbation'] != 'No Perturbation')]

g = sns.catplot(
    data=df2, x="Prompt", y="Accuracy", col="Perturbation",
    hue="Shots",
    kind="bar", height=6, aspect=0.5, palette="dark", alpha=.6,
)

df2["Accuracy"] = df2["Accuracy"] - df[(df["Prompt"] == "0CoT")
                                       & (~df["perturb_exemplar"])].iloc[2, 7]
print(df2.sort_values(['Perturbation', 'Prompt']))

# turn off x labels
g.set_xlabels("")

# get figure of g
g.fig.supxlabel("Prompt")
df2 = df[(df["Prompt"] == "0CoT") & (~df["perturb_exemplar"])]

# turn off legend
g._legend.remove()

# get axes of g
axes = g.axes.flatten()
axes[0].axhline(df2.iloc[0, 7], ls='--', color='red')
axes[1].axhline(df2.iloc[1, 7], ls='--', color='red')
axes[2].axhline(df2.iloc[2, 7], ls='--', color='red')
axes[3].axhline(df2.iloc[3, 7], ls='--', color='red', label='0')

# legend out of plot
plt.legend(loc='center right', bbox_to_anchor=(
    1.5, 0.5), ncol=1, title='Shots')

plt.savefig('images/perturb_shots.png', bbox_inches='tight')
