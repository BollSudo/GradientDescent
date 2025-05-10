from abc import ABC, abstractmethod
from typing import Callable, cast, Collection

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TP MODULE OPTIMISATION A2

# CONSTANTES
XMIN, XMAX = -2, 2
YMIN, YMAX = -2, 2
delta = 0.1 # deimmension d'une maille carrée

class Fonction2DAbstract(ABC):
    @abstractmethod
    def __call__(self, x, y):
        ...
    @abstractmethod
    def grad(self, x, y):
        ...

class Fonction2D(Fonction2DAbstract):
    """La fonction à deux variables à étudier"""
    def __init__(self, expression: str, gradx: str, grady: str):
        self.expression = expression
        self._f: Callable = lambda x, y: eval(expression)
        self._gradx: Callable = lambda x, y: eval(gradx)
        self._grady: Callable = lambda x, y: eval(grady)

    def __call__(self, x, y):
        return self._f(x, y)

    def grad(self, x, y):
        return self._gradx(x, y), self._grady(x, y)


def _create_grid(fonction: Fonction2D) -> tuple:
    """
    Créer la grille du plan et les images de chaque point de cette grille
    :param fonction: la fonction à étudier
    :return: tuple contenant les images de chaque point de la grille (x, y, z)
    """
    x = np.arange(XMIN, XMAX + delta, delta)
    y = np.arange(YMIN, YMAX + delta, delta)
    # Initialisation de la grille du plan
    x_mesh, y_mesh = np.meshgrid(x, y)
    # Images de chaque point de la grille
    z_mesh = fonction(x_mesh, y_mesh)
    return x_mesh, y_mesh, z_mesh

def draw_3d(fonction: Fonction2D) -> tuple[plt.Figure, Axes3D]:
    """
    Trace la représentation 3D de la fonction à deux variables
    :param fonction: la fonction à étudier
    :return: une figure contenant la représentation 3D
    """
    fig, ax = plt.subplots()
    fig.suptitle(f"Representation 3D de {fonction.expression}")
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    x, y, z = _create_grid(fonction)
    ax.plot_surface(x, y, z, cmap='viridis') # couleur de la surface
    return fig, ax

def draw_courbes_niveau(fonction: Fonction2D, levels=Collection[float]) -> tuple[plt.Figure, plt.Axes]:
    """
    Trace les courbes de niveau de la fonction à deux variables
    :param levels: les valeurs des courbes de niveau
    :param fonction: la fonction à étudier
    :return: une figure contenant les courbes de niveau
    """
    fig, ax = plt.subplots()
    fig.suptitle(f"Courbes de niveau de {fonction.expression}")
    x, y, z = _create_grid(fonction)
    contour = ax.contour(x, y, z, levels=levels, cmap='viridis')
    ax.clabel(contour) # affichage des valeurs des courbes de niveau
    return fig, ax