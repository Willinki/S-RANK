###################################################################
#           RELATIVE EVALUATION OF CLUSTERING                     #
#           -    based on RANK ALGORITHM    -                     #
###################################################################
# si valuta la qualità del clustering secondo i rankings delle varaibili forniti
# i rankings vanno ottenuti con l'algoritmo SRANK (file SRANK_results.ipynb)
# - dataframe: i dati completi
# - rankings: dataframe contenente le variabili ordinate per importanza, 
#             ottenuto tramite SRANK
# - algo: [kmeans], [agglomerative], [meanshift] Tipo di algoritmo da valutare.
# - eval_indexSTR: [CH], [DB], [sil]. Tipo di indice da usare

import pandas as pd 
import numpy as np 
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 

def clustering_evaluation(dataframe, rankings, algo, eval_indexSTR): 
    #array con le feature su cui fare clustering
    feature_group = []

    #- index tiene il miglior valore dell'indice per ogni gruppo di features
    #- ncluster tiene il numero di cluster per cui si ha l'indice migliore
    #  per ogni gruppo di features
    #- feature group va da 0 a nfeatures-1, solo un indice ordinale per il 
    #  plotting
    bestof_group_index = []
    bestof_group_ncluster = []
    bestof_group_feature_group = [i for i in range(len(rankings.index))]

    for feature in rankings["feature"]:
        #contatore
        print("\rFeature: ", len(feature_group) + 1, "/", 
              len(rankings.index), end = "\r"),
        #aggiungo una feature al gruppo e prendo i dati
        feature_group.append(feature)
        data_section = dataframe[feature_group] 


        # per il gruppo di var vario il numero di cluster 
        # fra 1 e 20, per ognuno calcolo l'indice e lo aggiungo 
        # ad un array 
        for nClusters in range(2, 21):
            #contiene gli indici per ogni nCluster
            nCluster_scores = []
            
            #algoritmi possibili
            algorithm_dict = {
                                "kmeans" : KMeans(n_clusters = nClusters, 
                                                  random_state = 1), 
                                'agglomerative' : AgglomerativeClustering(n_clusters = nClusters,
                                                                          affinity = "euclidean", 
                                                                          linkage = "ward"),
                                'meanshift' : MeanShift()
                             }

            #fitto e calcolo l'indice, lo aggiungo all'array
            model = algorithm_dict[algo]
            model.fit(data_section)
            labels = model.labels_
            
            #indici di performance possibili
            if eval_indexSTR == "CH":
                try:
                    eval_index = metrics.calinski_harabasz_score(data_section, labels)
                except:
                    eval_index = -2
            if eval_indexSTR == "DB":
                try:
                    eval_index = metrics.davies_bouldin_score(data_section, labels)
                except:
                    eval_index = -2
            if eval_indexSTR == "sil":
                try: 
                    eval_index = metrics.silhouette_score(data_section, labels, metric='l2')
                except:
                    eval_index = -2
            nCluster_scores.append(eval_index)

            #se l'algoritmo scelto è meanshift, non si deve ripetere il ciclo
            if algo == "meanshift":
                break
        
        #salvo l'indice migliore, e il relativo numero di clusters
        if eval_indexSTR == "DB":
            best_index = min(nCluster_scores)
        else:
            best_index = max(nCluster_scores)
        nClusters_best = nCluster_scores.index(best_index) + 2
        bestof_group_index.append(best_index)
        bestof_group_ncluster.append(nClusters_best)

    #CREO UN DATAFRAME COI DATI
    results = pd.DataFrame({
                                "index" : bestof_group_index, 
                                "n_clusters" : bestof_group_ncluster, 
                                "feature_group" : bestof_group_feature_group
                            })
    
    #STAMPO UN RIASSUNTO
    if eval_indexSTR == "DB":
        best_index = min(bestof_group_index)
    else:
        best_index = max(bestof_group_index)
    nClusters_best = bestof_group_index.index(best_index) + 2
    print("\r Best value of {} index: ".format(eval_indexSTR), best_index, "\n", 
          "Number of clusters for best value: ", nClusters_best, "\n",
          "Number of clusters: ", bestof_group_ncluster, end = "\r")
    
    #PLOTTING 
    plt.figure(figsize = (8, 5))
    mpl.style.use('fivethirtyeight')
    sns.set(rc = {
                    'axes.facecolor':'lightgray',
                    'axes.edgecolor': 'lightgray',
                    'figure.facecolor':'lightgray',
                    'axes.labelcolor': '#414141',
                    'text.color': '#414141',
                    'xtick.color': '#414141',
                    'ytick.color': '#414141',
                    'grid.color': 'ghostwhite',
                })
    chart = sns.lineplot(x = "feature_group" ,
                         y =  "index", 
                         data = results, 
                         color = "royalblue",
                         marker = 'o')
    plt.title("Performance [{}, {} index]".format(algo, eval_indexSTR), 
              loc = "left",
              fontdict = {'fontsize': 23,
                          'fontweight' : "semibold",
                          'verticalalignment': 'baseline',
                          'horizontalalignment': "left"})
    plt.xlabel("Feature group", fontsize = 15, color='#414141')
    plt.ylabel("{} Index value".format(eval_indexSTR), 
               fontsize = 15, color='#414141')
    plt.draw()
    plt.show()