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
import copy

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
    
def crossval_strat(X,Y,n_iterations,iteration):
    
    unique_Y=np.unique(Y)
    index_list=[]
    for i in range(len(unique_Y)):
        indexs, = np.where(Y == unique_Y[i])
        index_list.append(indexs)
    
    index_test_list=[]
    for i in range(len(index_list)):
        index_i=index_list[i]
        indexs_test=index_i[iteration*(len(index_i) // n_iterations): (iteration+1)*len(index_i)//n_iterations]
        index_test_list.append(indexs_test)

    X_Test=X[index_test_list[0]]
    for i in range(1,len(index_test_list)):
        X_Test=np.concatenate((X_Test,X[index_test_list[i]]))
     
    Y_Test=Y[index_test_list[0]]
    for i in range(1,len(index_test_list)):
        Y_Test=np.concatenate((Y_Test,Y[index_test_list[i]]))
    
    index_train=[]
    for i in range(len(X)):
        is_in=False
        for index in index_test_list:
            if i in index:
                is_in=True
    
        if is_in==False:
            index_train.append(i)
    
    X_Train=X[index_train]
    Y_Train=Y[index_train]
    
    return X_Train,Y_Train,X_Test,Y_Test

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    for i in range(nb_iter):
        newC = copy.deepcopy(C)
        desc_train,label_train,desc_test,label_test=crossval_strat(X,Y,nb_iter,i)
        newC.train(desc_train,label_train)
        acc_i=newC.accuracy(desc_test,label_test)
        perf.append(acc_i)
        print(i,": taille app.= ",label_train.shape[0],"taille test= ",label_test.shape[0],"Accuracy:",acc_i)
    
    ########################## COMPLETER ICI 
    
    
    ##########################
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)