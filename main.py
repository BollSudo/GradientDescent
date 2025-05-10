import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from algo import Fonction2D, draw_3d, draw_courbes_niveau, descente, print_result, draw_descente_in_3d, \
    draw_descente_in_courbes_niveau, mc_gradient, mc_calc_theorie, draw_results_mc, print_result_mc, \
    print_result_theorie

matplotlib.use('TkAgg', force=False) # Pour les graphes intéractifs si le moteur graphique backend est installé

# TP MODULE OPTIMISATION A2
def main():
    # partie3()
    partie4()

# MES FONCTIONS
F1 = Fonction2D("x**2+y**2", "2*x", "2*y")
F2 = Fonction2D("(x**2+y**2)/(x**2+1)", "-(2*x*(y**2-1))/((x**2+1)**2)", "2*y/(x**2+1)")
F3 = Fonction2D("(x**2-1)**2+(y**2-1)**2", "4*x*(x**2-1)", "4*y*(y**2-1)")
F4 = Fonction2D("np.cos(4*x)+np.sin(4*y)", "-4*np.sin(4*x)", "4*np.cos(4*y)")
F5 = Fonction2D("x**2*np.cos(4*x)+y**2*np.sin(4*y)", "2*x*(np.cos(4*x)-2*x*np.sin(4*x))", "2*y*(np.sin(4*y)+2*y*np.cos(4*y))")
F6 = Fonction2D("(x**2 + y)/(x**2+y**2+1)", "(2*x*(y**2-y+1))/((x**2+y**2+1)**2)", "((x**2)*(1-2*y)+1-y**2)/((x**2+y**2+1)**2)")
F7 = Fonction2D("(np.cos(2*x)**2)*np.cos(2*y)", "-2*np.sin(4*x)*np.cos(2*y)", "-2*np.sin(2*y)*(np.cos(2*x)**2)")

def partie3()-> None:
    """
    Utiliser l'algorithme pour minimiser des fonctions et observer le comportement de l'algorithme
    """
    # INIT (à configurer ici)
    x_init, y_init = 0.2, 0.7
    step = 0.1
    epsilon = 0.01
    n_max = 100
    levels = np.arange(-3, 3, 0.2) # [0.1,0.3,0.5,1,1.5,2,2.5,3]
    f = F7

    # MAIN
    fig3d, ax3d = draw_3d(f)
    fig2d, ax2d = draw_courbes_niveau(f, levels=levels)
    result = descente(f, x_init, y_init, step, epsilon, n_max)
    print_result(result)
    draw_descente_in_3d(ax3d, result)
    draw_descente_in_courbes_niveau(ax2d, result)
    plt.show()

def partie4()-> None:
    """
    Utiliser l'algorithme de descente de gradient pour retrouver les moindres carrés
    """
    # INIT (à configurer ici)
    # x = [1, 2, 3, 4]
    # y = [3, 3, 4, 6]
    x = [1, 2, 3, 4, 6, 7]
    y = [3, 3, 4, 6, 6, 8]
    step = 0.1
    epsilon = 0.01
    n_max = 100

    # MAIN
    resultmc = mc_gradient(x, y, len(x), step, epsilon, n_max)
    resultmc_theo = mc_calc_theorie(x, y)
    draw_results_mc(x, y, resultmc, resultmc_theo)
    print_result_mc(resultmc)
    print_result_theorie(resultmc_theo)

    plt.show()

if __name__ == '__main__':
    main()