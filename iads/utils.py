# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 


def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data_desc =np.random.uniform(binf,bsup,(n*p,p)) 
    data_label =np.asarray([-1 for i in range(n*p//2)] + [+1 for i in range(n*p//2)])
    return data_desc,data_label

    raise NotImplementedError("Please Implement this method")

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # On tire aléatoirement tous les exemples des classes -1 et +1
    negative_desc_valeur = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    positive_desc_valeur = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    
    # On fusionne les deux ensemble pour obtenir descriptions desc=[valeur négative,valeur positive]
    desc = np.vstack((negative_desc_valeur, positive_desc_valeur))
    
    # On crée les labels
    labels = np.asarray([-1 for i in range(nb_points)] + [+1 for i in range(nb_points)])
    
    return desc, labels
    raise NotImplementedError("Please Implement this method")

def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
   #TODO: A Compléter
    # On sépare les exemples de chaque classe
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    
    # Affichage des deux arrays obtenus
    plt.scatter(data_negatifs[:,0], data_negatifs[:,1], marker='o',color="red")
    plt.scatter(data_positifs[:,0], data_positifs[:,1], marker='x',color="blue")
    plt.show()
    #raise NotImplementedError("Please Implement this method")

def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])