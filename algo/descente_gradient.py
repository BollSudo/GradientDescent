from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from algo.fonction_2d import Fonction2DAbstract

# TP MODULE OPTIMISATION A2

def descente(
        f: Fonction2DAbstract,
        x_init: float,
        y_init: float,
        step: float,
        epsilon: float,
        n_max: int
        ) -> tuple:
    """Applique l'algorithme de descente de gradient à la fonction f en partant de (x_init, y_init)
    et avec un pas de step, un epsilon et un nombre maximum de pas n_max."""
    square_epsilon = epsilon * epsilon
    x_array = [x_init]
    y_array = [y_init]
    z_array = [f(x_init, y_init)]
    gradient = f.grad(x_init, y_init)
    square_norme_grad = gradient[0] ** 2 + gradient[1] ** 2
    n = 0
    while n < n_max and square_norme_grad > square_epsilon:
        n += 1
        x = x_array[-1] - step * gradient[0]
        y = y_array[-1] - step * gradient[1]
        z = f(x, y)
        gradient = f.grad(x, y)
        square_norme_grad = gradient[0] ** 2 + gradient[1] ** 2

        x_array.append(x)
        y_array.append(y)
        z_array.append(z)
    return x_array, y_array, z_array, n

def print_result(result_descente: tuple):
    """Affiche le résultat de la descente de gradient entré en paramètre dans la console."""
    a, b, c, n = result_descente
    print(f"x = {a[-1]}\ny = {b[-1]}\nz = {c[-1]}\nn = {n}")

def draw_descente_in_3d(ax: Axes3D, result_descente: tuple):
    """Ajoute la trajectoire de la descente de gradient dans la représentation 3D entré en paramètre."""
    a, b, c, n = result_descente
    ax.plot(a, b, c, 'ro', linewidth=2)

def draw_descente_in_courbes_niveau(ax: Axes, result_descente: tuple):
    """Ajoute la trajectoire de la descente de gradient dans la représentation des courbes de niveau entré en paramètre."""
    a, b, c, n = result_descente
    ax.plot(a[1:-1], b[1:-1], 'r+:', ms=3, linewidth=1)
    ax.plot(a[0], b[0], 'ro:')
    ax.plot(a[-1], b[-1], 'rx:')