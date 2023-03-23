# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import math

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne=sum(L)/len(L)
    
    ecart_type=0
    for pref in L:
        ecart_type=ecart_type+((pref-moyenne)*(pref-moyenne))
        
    return (moyenne,math.sqrt(ecart_type/len(L)))
    raise NotImplementedError("Vous devez implémenter cette fonction !")    