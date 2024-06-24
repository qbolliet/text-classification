# Importation des modules
# Modules de base
import pandas as pd
# Modules graphiques
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
# Nuages de mots
from wordcloud import WordCloud

# Fonction construisant une distribution
def build_hisplot(data : pd.Series, title : str, xlabel : str, ylabel : str) -> None :
    """
    Build and display a histogram plot with a KDE (Kernel Density Estimate) curve.

    Parameters:
        data (pd.Series): The data series to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    # Initialisation de la figure
    plt.figure(figsize=(10, 6))
    # Création du graphe
    sns.histplot(data=data, kde=True)
    # Titre du graphe
    plt.title(title)
    # Titre des axes
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.show()

# Fonction construisant un diagramme en barres
def build_barplot(data : pd.Series, title : str, xlabel : str, ylabel : str) -> None :
    """
    Build and display a bar plot.

    Parameters:
        data (pd.Series): The data series to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    # Initialisation de la figure
    plt.figure(figsize=(10, 6))
    # Création du graphe
    sns.countplot(x=data)
    # Titre du graphe
    plt.title(title)
    # Titre des axes
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.show()

def build_wordcloud(text : str, title : str) -> None :
    """
    Build and display a word cloud.

    Parameters:
        text (str): The text from which to generate the word cloud.
        title (str): The title of the plot.

    Returns:
        None
    """
    # Generation d'une nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text=text)

    # Initialisation de la figure
    plt.figure(figsize=(10, 5))
    # Construction du graphique
    plt.imshow(wordcloud, interpolation='bilinear')
    # Suppression des axes
    plt.axis('off')
    # Titre du graphique
    plt.title(title)

    plt.show()