import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_singleVar_numeric(df, col, xlim=None, ylim=None):
    plt.figure(figsize=(18, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, bins=50)
    plt.title(f'Distribuição: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequencia')

    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col].dropna())
    plt.title(f'BoxPlot: {col}')
    plt.ylabel(col)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()

def plot_singleVar_categorical(df, col):
    descending_order = df[col].value_counts().index

    plt.figure(figsize=(18, 6))
    ax = sns.countplot(x=col, data=df, palette='viridis', hue=col, legend=False, order=descending_order)
    plt.title(f'Distribuição: {col}')
    plt.xlabel(col)
    plt.ylabel('Contagem')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

def plot_against_target_categorical(df, col, target):
    descending_order = df.groupby(col)[target].median().sort_values().index

    plt.figure(figsize=(18, 6))
    sns.boxplot(x=col, y=target, data=df, palette='viridis', hue=col, legend=False, order=descending_order)
    plt.title(f'{target} por {col}')
    plt.xlabel(col)
    plt.ylabel(target)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_against_target_numerical(df, col, target):
    plt.figure(figsize=(18, 6))
    sns.scatterplot(x=col, y=target, data=df, alpha=0.6)

    # Calculate correlation and p-value
    df_cleaned = df[[col, target]].dropna()
    if not df_cleaned.empty:
        correlation, p_value = pearsonr(df_cleaned[col], df_cleaned[target])
        annotation_text = f'Correlação (Pearson): {correlation:.2f}\nP-valor: {p_value:.3f}'
        plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9))
    else:
        plt.annotate('Não foi possível calcular a correlação', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.5))

    plt.title(f'{target} vs {col}')
    plt.xlabel(col)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()
