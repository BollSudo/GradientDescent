import numpy as np
from matplotlib import pyplot as plt

from algo.fonction_2d import Fonction2DAbstract
from algo.descente_gradient import descente

class FonctionMinimisation(Fonction2DAbstract):
    """La fonction de minimisation des moindres carrés, pour une liste de points (x_i, y_i) contenu dans les variables x et y."""
    def __init__(self, x, y, n: int):
        self._x = np.array(x)
        self._y = np.array(y)
        self._n = n
        self._f = lambda a, b: 1/n * np.sum((self._y - a*self._x - b)**2)

    def __call__(self, a, b):
        return self._f(a, b)

    def grad(self, a, b):
        grad_a = 1 / self._n * np.sum(1 / self._n * np.sum(-2 * self._x * (self._y - a * self._x - b)))
        grad_b = 1 / self._n * np.sum(1 / self._n * np.sum(-2 * (self._y - a * self._x - b)))
        return grad_a, grad_b


def mc_gradient(x, y, n: int, step: float, epsilon: float, n_max: int) -> tuple:
    """Applique l'algorithme de descente de gradient à la fonction de minimisation des moindres carrés."""
    f = FonctionMinimisation(x, y, n)
    return descente(f, 0, 0, step, epsilon, n_max)

def mc_calc_theorie(x, y) -> tuple:
    """Calcule les résultats théoriques de la méthode des moindres carrés (y fonction de x)."""
    cov_matrice = np.cov(np.array(x), np.array(y))
    a = cov_matrice[0][1] / cov_matrice[0][0]
    b = np.mean(y) - a * np.mean(x)
    e = cov_matrice[1][1] - (cov_matrice[0][1] ** 2) / cov_matrice[0][0]
    r_theo = cov_matrice[0][1] / np.sqrt(cov_matrice[0][0] * cov_matrice[1][1])
    return a, b, e, r_theo

def print_result_mc(result):
    """Affiche les résultats dans la console."""
    a, b, e, n = result
    print(f"a = {a[-1]}\nb = {b[-1]}\ne = {e[-1]}\nn = {n}")

def print_result_theorie(result):
    """Affiche les résultats théoriques dans la console."""
    a, b, e, r_theo = result
    print(f"a_theo = {a}\nb_theo = {b}\ne_theo = {e}\nr_theo = {r_theo}")

def draw_results_mc(x, y, result, result_theo = None) -> tuple[plt.Figure, plt.Axes]:
    """Dessine le nuage de points et la droite de regression linéaire obtenu par la descente de gradient."""
    fig, ax = plt.subplots()
    ax.grid()
    fig.suptitle("Régression linéaire par descente du gradient")

    x_line = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)
    a, b, e, n = result
    y_line = a[-1] * x_line + b[-1]

    ax.scatter(x, y, c='r', marker='+', s=50, zorder=3)
    ax.plot(x_line, y_line, 'b-', ms=3, linewidth=2, zorder=3, label="algo")

    if result_theo is not None:
        a_theo, b_theo, e_theo, r_theo = result_theo
        y_line_theo = a_theo * x_line + b_theo
        ax.plot(x_line, y_line_theo, 'r--', ms=3, linewidth=1, zorder=3, label="theorie")

    ax.legend()
    return fig, ax
