import pandas as pd              # Manipulação de dados (DataFrame)
import seaborn as sns           # Gráficos estatísticos
import matplotlib.pyplot as plt # Plotagem com Matplotlib
import numpy as np              # Operações numéricas (máscara do heatmap)

# 1 - Ler o CSV com os dados médicos
df = pd.read_csv("medical_examination.csv")

# 2 - Criar coluna 'overweight' via IMC (peso / altura^2); IMC>25 => 1 (acima do peso), senão 0
df["overweight"] = ((df["weight"] / ((df["height"] / 100) ** 2)) > 25).astype(int)

# 3 - Normalizar colesterol e glicose: 0 = bom (valor 1), 1 = ruim (valores >1)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


# 4 - Função que gera o gráfico categórico (catplot)
def draw_cat_plot():
    # 5 - Converter para formato longo (melt) as variáveis categóricas, mantendo 'cardio'
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # 6 - Agrupar por cardio/variável/valor e contar ocorrências (renomear contagem para 'total')
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )
    
    # 7 - Criar gráfico de barras por variável, separado por cardio, com hue=valor (0 bom / 1 ruim)
    g = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    )

    # 8 - Obter a figura do FacetGrid para salvar/retornar
    fig = g.fig

    # 9 - Salvar a figura e retornar
    fig.savefig("catplot.png")
    return fig


# 10 - Função que gera o mapa de calor (heatmap) das correlações
def draw_heat_map():
    # 11 - Limpar dados: manter (ap_lo<=ap_hi) e remover outliers por percentis de altura/peso
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12 - Calcular a matriz de correlação das variáveis numéricas
    corr = df_heat.corr()

    # 13 - Criar máscara para ocultar o triângulo superior (evita duplicação visual)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Configurar a figura e o eixo do Matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 - Plotar o heatmap com anotações, grade quadrada e barra de cor reduzida
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    # 16 - Salvar a figura e retornar
    fig.savefig("heatmap.png")
    return fig
