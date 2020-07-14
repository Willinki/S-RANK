########################################################################
#           DATAFRAME EXTENDED                                         #
########################################################################
#the structure is an extension of the pandas.Dataframe structure.
#it contains additional information such as the list of binary variables, 
#non-numerical variables, the matrix of distances and similarities.
#also, given a dataset on construction, ot perfomrs standardization and
#removal of outliers
import numpy as np
import pandas as pd 
import math
import sys
from sklearn import preprocessing
from scipy import stats
class dataframe_ext:
    #attributes: 
    #   - dataFrame:                df
    #   - type of varaibles:        TYPE                     
    #   - discrete variables        discrete_vars -> None if TYPE = continous
    #                                                all if TYPE = discrete
    #                                                list provided if TYPE = mixed
    #   - continous variables       continous_vars -> same logic as discrete
    #   - distance type:            dist          -> euclidean if TYPE = continous
    #                                                hamming otherwise
    #   - distances matrix:         D
    #   - similarities matrix:      S
    #   - similarity parameter:     alpha
    #   - entropy value:            E

    #########################################
    # CONSTRUCTOR - most of the attributes  #
    # are actually blank until complete     #
    # is called                             #
    #########################################
    def __init__(self, dataframe, vars_type, discrete_vars_list = None):
        #the data is stored
        if type(dataframe) == pd.core.frame.DataFrame:
            self.df = dataframe
        else: 
            sys.exit("Please specify a valid dataframe")
            
        #according to the type of the variables, each attribute is set
        if vars_type not in ["continous", "mixed", "discrete"]:
            sys.exit("Please specify a valid vars_type: [continous, discrete, mixed]")
        elif vars_type == "continous":
            self.TYPE = "continous"
            self.dist = self.euclidean_distance #TODO
            self.discrete_vars = []
            self.continous_vars = self.dataframe.columns
        elif vars_type == "discrete":
            self.TYPE = "discrete"
            self.dist = self.hamming_distance #TODO
            self.discrete_vars = self.dataframe.columns 
            self.continous_vars = []
        elif vars_type == "mixed":
            self.TYPE = "mixed"
            self.dist = self.hamming_distance #TODO
            if type(discrete_vars_list) != list:
                sys.exit("Please provide a valid discrete variable list")
            else:
                try: 
                    _ = self.dataframe[discrete_vars_list]
                except:
                    sys.exit("Columns in discrete_var_list are not in the Dataframe")
                self.discrete_vars = discrete_vars_list
                self.continous_vars = self.dataframe.drop(columns = self.discrete_vars).columns
                #TODO FARE DISCRETIZZAZIONE DELLE VARIABILI CONTINUE
        
        #all the other parameters are set to None
        self.D = None
        self.S = None
        self.alpha = None
        self.E = None

    #########################################
    # COMPLETE - determines the value of    #
    # all the other attributes              #
    #########################################
    def complete(self):                       
        self.dist_matrix()
        self.set_alpha()
        self.sim_matrix()
        self.simmatrix_entropy()
    
    #########################################
    # CLEAN - removes all rows that have    #
    # outliers in them, i.e. values outside #
    # the interval mean +- 3*devstd         #
    #########################################
    #pulizia del dataset dagli outliers, non variabili binarie
    def clean(self):
        ZS = stats.zscore(self.df.drop(columns = self.binary_vars))
        self.df = self.df[(np.abs(ZS) < 3).all(axis = 1)]
        return self
    
    #########################################
    # STANDARDIZE - selects only continues  #
    # variables and performs gaussian       #
    # standardization                       #
    #########################################
    def standardize(self):
        df_binary = self.df[self.binary_vars] 
        df_nonbinary = self.df.drop(columns = self.binary_vars)
        nonbinary_vars = df_nonbinary.columns
        scaler = preprocessing.StandardScaler()
        df_nonbinary = pd.DataFrame(
                                        scaler.fit_transform(df_nonbinary),
                                        columns = nonbinary_vars
                                   )
        self.df = df_nonbinary.join(df_binary)
        return self

    #euclidean distance between continous attributes
    @staticmethod 
    def euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    #funzione per il calcolo della matrice 
    #delle distanze
    def dist_matrix(self):
        #se il dataframe viene da un sample
        #funziona comunque
        df = self.df.reset_index(drop = True)
        self.D = np.asmatrix(
                    [
                        [
                            self.dist_measure(row1, row2) 
                            for _, row1 in df.iterrows()
                        ]
                        for _, row2 in df.iterrows()
                    ]
                )
        self.D = np.asmatrix(self.D)
    
    #funzione per il calcolo di alpha
    #richiede D
    def set_alpha(self):
        self.alpha = -math.log(0.5) / np.matrix.mean(self.D)

    #converte una distanza in similarità 
    #richiede alpha
    def sim(self, dist_value):
        return math.exp(- self.alpha * dist_value)

    #funzione per il calcolo della matrice 
    #delle similarità (richiede D e alpha)
    def sim_matrix(self):
        #vettorizzo la funzione di similarità 
        vsim = np.vectorize(self.sim)
        #calcolo la similarità fra ogni elemento
        self.S = vsim(self.D)
    
    #converte un valore di similarità
    #in entropia
    @staticmethod
    def sim_to_entropy(sv):
        if sv != 1:
            return - ((sv * math.log2(sv)) + ((1-sv)*math.log2(1-sv)))
        else:
            return float('nan')
    
    #funzione per il calcolo dell'entropia associata
    #al dataset
    def simmatrix_entropy(self):
        entropies = [self.sim_to_entropy(x) for x in np.nditer(self.S)]
        self.E = np.nansum(entropies)/2 #FLOAT
