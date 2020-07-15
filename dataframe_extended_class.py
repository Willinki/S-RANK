########################################################################
#           DATAFRAME EXTENDED                                         #
########################################################################
#the structure is an extension of the pandas.Dataframe structure.
#it contains additional information such as the list of numerical variables, 
#non-numerical variables, the matrix of distances and similarities.
#also, given a dataset on construction, ot performs rescaling, discretization and
#removal of outliers
import numpy as np
import pandas as pd 
import math
import sys
from sklearn import preprocessing
from scipy import stats
class dataframe_ext:
    #########################################
    # Attributes: 
    #   - dataFrame:                df
    #   - type of varaibles:        TYPE                     
    #   - discrete variables        discrete_vars -> None if TYPE = continous
    #                                                all if TYPE = discrete
    #                                                list provided if TYPE = mixed
    #   - continous variables       continous_vars -> same logic as discrete
    #   - object variables          object_vars -> variables of type object
    #   - distance type:            dist          -> euclidean if TYPE = continous
    #                                                hamming otherwise
    #   - distances matrix:         D
    #   - similarities matrix:      S
    #   - similarity parameter:     alpha
    #   - entropy value:            E
    #########################################

    #########################################
    # CONSTRUCTOR - most of the attributes  #
    # are actually blank until complete     #
    # is called                             #
    #########################################
    def __init__(self, dataframe, vars_type, discrete_vars_list = None, clean = False, rescale = False):
        #the data is stored
        if type(dataframe) == pd.core.frame.DataFrame:
            self.df = dataframe
        else: 
            sys.exit("Please specify a valid dataframe")
            
        #first we set the list of object_vars
        self.object_vars = [x for x in self.df.columns if self.df[x].dtype == "object"]
        #then a little control over inferred types
        if len(self.object_vars) != 0 & vars_type == "continous":
            sys.exit("vars_type is set to continous but object type columns are present.\
                      If there are non numerical variables in the dataset please set \
                      vars_type = mixed, otherwise make sure that dtypes in the \
                      dataframe are inferred correctly.")

        #according to vars_type, every parameter is set
        if vars_type not in ["continous", "mixed", "discrete"]:
            sys.exit("Please specify a valid vars_type: [continous, discrete, mixed]")
        elif vars_type == "continous":
            self.TYPE = "continous"
            self.dist = self.euclidean_distance 
            self.discrete_vars = []
            self.continous_vars = self.df.drop(columns = self.object_vars)
            if clean == True:
                self.clean()
            if rescale == True:
                self.rescale()
        elif vars_type == "discrete":
            self.TYPE = "discrete"
            self.dist = self.hamming_distance 
            self.discrete_vars = self.df.drop(columns = self.object_vars) 
            self.continous_vars = []
            if clean == True:
                self.clean()
            if rescale == True:
                self.rescale()
        elif vars_type == "mixed":
            self.TYPE = "mixed"
            self.dist = self.hamming_distance
            if type(discrete_vars_list) != list:
                sys.exit("Please provide a valid discrete variable list")
            else:
                try: 
                    _ = self.dataframe[discrete_vars_list]
                except:
                    sys.exit("Columns in discrete_var_list are not in the Dataframe")
                self.discrete_vars = discrete_vars_list
                self.continous_vars = (self.dataframe
                                           .drop(columns = [self.discrete_vars, self.object_vars])
                                           .columns)
                if clean == True:
                    self.clean()
                if rescale == True:
                    self.rescale()
                self[self.continous_vars].discretize()
        
        #all the other parameters are set to None
        self.D = None
        self.S = None
        self.alpha = None
        self.E = None

    #########################################
    # COMPLETE - determines the value of    #
    # all the other attributes in the right #
    # order                                 #
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
    def clean(self):
        ZS = stats.zscore(self.df[self.continous_vars])
        self.df = self.df[(np.abs(ZS) < 3).all(axis = 1)]
        return self
    
    #########################################
    # RESCALE - Rescale variables to have   #
    # values from 0 to 1                    #
    #                                       #
    #########################################
    def rescale(self):
        scaler = preprocessing.MinMaxScaler()
        self.df = pd.DataFrame(
                                scaler.fit_transform(self.df),
                                columns = self.df.columns
                              )
        return self
    
    #########################################
    # DISCRETIZE - makes all the variables  #
    # discrete (could have guessed that)    #
    #                                       #
    #########################################
    def discretize(self):
        disc = preprocessing.KBinsDiscretizer(n_bins = 10, 
                                              encode = "ordinal", 
                                              strategy = "uniform")
        self.df = pd.DataFrame(
                               disc.fit_transform(self.df), 
                               columns = self.df.columns
                              )
        return self
    

    #########################################
    # EUCLIDEAN DISTANCE - the distance     #
    # used if the data contains only        #
    # numerical attriubtes                  #
    #########################################
    @staticmethod 
    def euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    #########################################
    # HAMMING DISTANCE - the distance       #
    # used if the data contains at least    #
    # one discrete attribute                #
    #########################################
    @staticmethod
    def hamming_distance(x1, x2):
        dist = 0
        for x1i, x2i in zip(x1, x2):
            if x1i != x2i:
                dist = dist + 1
        return dist / len(x1)
    
    #########################################
    # DIST_MATRIX - calculates the distance #
    # matrix for the data                   #
    #                                       #
    #########################################
    def dist_matrix(self):
        #this is to handle the fact that the dataframe
        #could be a sample of a larger one
        df = self.df.reset_index(drop = True)
        self.D = np.asmatrix(
                    [
                        [
                            self.dist(row1, row2) 
                            for _, row1 in df.iterrows()
                        ]
                        for _, row2 in df.iterrows()
                    ]
                )
    
    #########################################
    # SET_ALPHA - sets the parameter for    #
    # the computation of similarity         #
    # requires D to be set                  #
    #########################################
    def set_alpha(self):
        self.alpha = -math.log(0.5) / np.matrix.mean(self.D)

    #########################################
    # SIM - converts a distance measure in  #
    # a similarity one                      #
    # requires alpha                        #
    #########################################
    def sim(self, dist_value):
        return math.exp(- self.alpha * dist_value)

    #########################################
    # SIM_MATRIX - calculates the           #
    # similarity matrix                     #
    #                                       #
    #########################################
    def sim_matrix(self):
        #vectorizing sim  
        vsim = np.vectorize(self.sim)
        #setting the similarity matrix
        self.S = vsim(self.D)
    
    #########################################
    # SIM_TO_ENTROPY - convert a similarity #
    # value into an entropy                 #
    #                                       #
    #########################################
    @staticmethod
    def sim_to_entropy(sv):
        return - ((sv * math.log2(sv)) + ((1-sv)*math.log2(1-sv)))
            
    
    #########################################
    # SIMMATRIX_ENTROPY - calculates the    #
    # entropy of the dataset                #
    #                                       #
    #########################################
    def simmatrix_entropy(self):
        entropies = [self.sim_to_entropy(x) for x in np.nditer(self.S)]
        self.E = np.nansum(entropies)/2 #FLOAT
