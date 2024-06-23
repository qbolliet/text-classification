# Importation des modules
# Modules de base
import os
import pandas as pd
from typing import Union
# Modules graphiques
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Fonction de définition de la chrte igf
def set_igf_style() -> None:
    """Sets the default style for all graphs according to igf chart"""

    # Installation de la police Cambria
    fontManager.addfont(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cambria.ttf"))
    #fontManager.addfont("Cambria.ttf")

    # Figure size
    plt.rcParams['figure.figsize'] = (15, 7)

    # Line plot styles
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

    # Axis labels and ticks
    plt.rcParams['font.family'] = 'Cambria'
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    # Legend
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.title_fontsize'] = 16
    plt.rcParams['legend.framealpha'] = 0

    plt.rcParams['legend.loc'] = 'upper center'

    # Remove top and right spines
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True

    # Set custom colormap
    plt.rcParams['axes.prop_cycle'] = cycler('color', ["#096c45", "#737c24", "#d69a00", "#e17d18", "#9f0025", "#ae535c"])


# Fonction définissant un ScalarMappable à partir de la ColorMap de l'IGF en la mettant à l'échelle du min et du max
def get_scalar_mappable(data : Union[pd.Series, pd.DataFrame]) -> ScalarMappable :
    # Définition de la ColorMap de l'IGF
    cmap_igf = LinearSegmentedColormap.from_list("charte", ["#096c45", "#737c24", "#d69a00", "#e17d18", "#9f0025"], N=256)

    # Définition des valeurs minimales et maximales des données
    value_min = data.min()
    value_max = data.max() 
    # Normalisation des valeurs de la ColorMap
    norm = Normalize(vmin=value_min, vmax=value_max)
    # Création du ScalarMappable
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap_igf)

    return scalar_mappable