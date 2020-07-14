from utility_scripts.dataframe_extended_class import dataframe_ext
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools

#dato un dataframe standardizzato (si suppone) e il ranking delle variabili
#la funzione restituisce un plot dei punti ridimensionati attraverso manifold 
#learning
# -dataframe: i punti, possibilmente provenienti da dataframe_ext dopo standardize e clean
# -rankings: tabella proveniente da SRANK-results
class manifold_representation:

    def __init__(self, dataframe, ranking):
        self.dataframe = dataframe
        self.ranking = ranking["feature"].values
        self.manifolds = None

    def manifold_plots(self):
        #################################
        # CALCOLO DEI MANIFOLD          #
        #################################
        self.manifolds = []
        feature_list = []
        for feature in self.ranking:
            print("\rFeature number: ", len(feature_list)+1, 
                  "/ ", len(self.ranking), end = "\r")
            feature_list.append(feature)
            if len(self.feature_list) <= 2:
                continue
            points = self.dataframe[feature_list].to_numpy()
            sheet = TSNE(n_components = 2).fit_transform(points)
            self.manifolds.append(sheet)
        
        #################################
        # SUBPLOTS PER OGNI MANIFOLD    #
        #################################
        n_plots = len(feature_list) - 2
        ncols = 4
        nrows = (n_plots % ncols) + ncols
        cm = np.array([x for x in range(nplots)]).reshape((nrows,ncols))
        sns.set(style="white", palette="muted", color_codes=True)
        f, axes = plt.subplots(nrows = nrows, ncols = ncols, 
                            figsize = (5*nrows, 5*ncols), 
                            sharex = True, sharey = True)

        for i, j in itertools.product(range(nrows), range(ncols)):
            points = manifolds[cm[i, j]]
            points = pd.DataFrame({
                                    "x" : points[:,0],
                                    "y" : points[:,1]
                                })
            sns.scatterplot(x = "x", y = "y", data = points, ax = axes[i, j])
            axes[i, j].title.set_text("N_features: {}".format(cm[i, j]+3))