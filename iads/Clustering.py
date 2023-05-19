# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist

# ------------------------ 

def normalisation(df):
    result = df.copy()
    for col in df.columns:
        max_value = max(df[col])
        min_value = min(df[col])
        result[col] = (df[col] - min_value) / (max_value - min_value)
    return result

def dist_euclidienne(x,y):
    return np.linalg.norm(x-y)

def centroide(X):
    return np.mean(X,axis=0)

def dist_centroides(X,Y):
    return dist_euclidienne(centroide(X),centroide(Y))

def initialise_CHA(df):
    return {i:[i] for i in range(len(df))}


def fusionne(df, partition, verbose=False):
    dist_min = +np.inf
    k1_dist_min, k2_dist_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist= dist_centroides(df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = [*partition[k1_dist_min], *partition[k2_dist_min]]
    if verbose and k1_dist_min !=-1:
        print("Distance mininimale trouvée entre ["+str(k1_dist_min) +"," +str(k2_dist_min) +"]  = "+str(dist_min))
    return p_new, k1_dist_min, k2_dist_min, dist_min

def CHA_centroid(df,verbose=False,dendrogramme=False):
    result = []
    partition = initialise_CHA(df)
    for o in range(len(df)):
        partition,k1, k2, distance = fusionne(df, partition,verbose)
        if k1!=-1 and k2 !=-1:
            result.append([k1, k2, distance, len(partition[max(partition.keys())])])
        
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(result, leaf_font_size=24.,)

        # Affichage du résultat obtenu:
        plt.show()
    return result

from scipy.spatial.distance import cdist

def dist_linkage(linkage, arr1, arr2):
    r = cdist(arr1,arr2, 'euclidean')
    if linkage == 'complete':
        return np.max(r)
    if linkage == 'simple':
        return np.min(r)
    if linkage == 'average':
        return np.mean(r)
    
def fusionne_linkage(df, linkage,partition, verbose=False):
    dist_min = +np.inf
    k1_dist_min, k2_dist_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist= dist_linkage(linkage,df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = [*partition[k1_dist_min], *partition[k2_dist_min]]
    if verbose and k1_dist_min !=-1:
        print("Distance mininimale trouvée entre ["+str(k1_dist_min) +"," +str(k2_dist_min) +"]  = "+str(dist_min))
    return p_new, k1_dist_min, k2_dist_min, dist_min

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    
    if linkage=='centroid':
        return CHA_centroid(DF,verbose,dendrogramme)
    
    result = []
    partition = initialise_CHA(DF)
    for o in range(len(DF)):
        partition,k1, k2, distance = fusionne_linkage(DF,linkage,partition,verbose)
        if k1 !=-1 and k2 != -1:
            result.append([k1, k2, distance, len(partition[max(partition.keys())])])
        
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(result, leaf_font_size=24.,)

        # Affichage du résultat obtenu:
        plt.show()
    return result


def CHA_complet(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'complete',verbose,dendrogramme)

def CHA_simple(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'simple',verbose,dendrogramme)

def CHA_average(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'average',verbose,dendrogramme)

def inertie_cluster(Ens):
    center=centroide(Ens)
    return sum(dist_euclidienne(center,v)**2 for v in np.array(Ens))

def init_kmeans(K,Ens):
    df_Ens=pd.DataFrame(Ens)
    return np.array(df_Ens.sample(n=K))

def plus_proche(Exe,Centres):
    return np.argmin([dist_euclidienne(Exe,centre) for centre in Centres])

def affecte_cluster(Base,Centres):
    dict_centre={i:[] for i in range(0,len(Centres))}
    for j in range (0,len(Base)):
        dict_centre[plus_proche(np.array(Base)[j],Centres)].append(j)
        
    return dict_centre

def nouveaux_centroides(Base,U):
    Base_numpy=np.array(Base) #pour pouvoir utiliser np.mean
    result=[]
    for k,elems in U.items():
        result.append(np.mean([Base_numpy[i] for i in elems],axis=0))
    
    return np.array(result)

def inertie_globale(Base, U):
    Base_numpy=np.array(Base)
    return sum([inertie_cluster([Base_numpy[i] for i in valeur]) for valeur in U.values()])

def kmoyennes(K, Base, epsilon, iter_max,affichage=True):
    Centres=init_kmeans(K,Base)
    U=affecte_cluster(Base,Centres)
    inertie_nouv=inertie_globale(Base,U)
    if affichage:
        print("Iteration : ",1," Inertie : ",inertie_nouv," Difference : ",inertie_nouv-epsilon-1)
    for i in range(1,iter_max):
        inertie_ancien=inertie_nouv
        #Recalcul de Centres et U
        Centres=nouveaux_centroides(Base,U)
        U=affecte_cluster(Base,Centres)
        inertie_nouv=inertie_globale(Base,U)
        diff=inertie_ancien-inertie_nouv
        if affichage:
            print("Iteration : ",i+1," Inertie : ",inertie_nouv," Difference : ",diff)
        if (diff < epsilon):
            break
        
    return Centres,U

def affiche_resultat(Base,Centres,Affect):
    couleurs = ["b","g","c","m","y","k","w"]
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
    for elem in Affect.values():
        c=np.random.choice(couleurs)
        for i in elem:
            plt.scatter(Base.iloc[i,0],Base.iloc[i,1],color=c)
            
def distance_max_cluster(cluster):
    return np.max(cdist(cluster, cluster))


def co_dist(X, U):
    d = 0
    X = np.array(X)
    for idxs in U.values():
        d += distance_max_cluster(X[idxs])
    return d


def index_Dunn(Base,Centres,U):
    return co_dist(Base, U) / inertie_globale(Base, U)