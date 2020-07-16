# S-RANK

![](https://img.shields.io/github/license/Willinki/S-RANK?color=blue&style=flat-square)

A basic implementation of the Scalable RANK Algorithm, for feature selection in unsupervised learning problems, as described [in this article](https://www.public.asu.edu/%7Ehuanliu/papers/pakdd00clu.pdf "Feature Selection for Clustering") by Manoranjan Dash and Huan Liu.

## Description

All the theoretical details are presented inside the article above. We implemented the RANK and SRANK algorithm following its indications.

## Usage

In order to use the algorithm, the repository must be first cloned. 
  ``` 
  git clone https://github.com/Willinki/S-RANK.git 
  ```

Then, to import:
  ```
  from S-RANK.FeatureRanker import SRANK
  ```
Then, in order to use the algorithm:
  ```
  ranker = SRANK()
  ranker.apply(df_big, vars_type, discrete_var_list = None, clean_bool = False, rescale_bool = False, 
               N_SAMPLE = 40, SAMPLE_SIZE = 50)
  ```
  
__Arguments:__
* ```df_big```: the whole data. It has to be a pandas.Dataframe object. Please make sure that all variable dtypes are inferred correctly. It is important to have numeric variables and non-numeric variables (dtype: object) inferred correctly. Up to now, the algorithm is able to handle numeric and object data, any other type (like datetimes) will not be treated. It is recommended to remove any non-numerical and non-object feature.
* ```vars_type``` :  the type of data in the dataframe. String that can take 3 values: ```continous``` if all the variables have continous numerical values. ```discrete``` if all the variables are categorical (encoded as string or number makes no difference), ```mixed``` if there's both.
* ```discrete_vars_list``` : needs to be set only if ```vars_type = "mixed" ```. It is a list containing the names of the categorical variable in the dataframe. If ```vars_type = "continous"``` or ```discrete``` this must be set to None.
* ```clean_bool``` if this option is set to true, outliers removal is performed. Any instance (row) is identified as an outlier if any of its feature lays outside the interval M +- 3 * s where M is the mean value of the feature and s is its standard deviation.
* ```rescale_bool``` if this option is set to True, rescaling of continous variables is performed, i.e., all values are rescaled to be inside the interval \[0;1\]. This affects only continous variables, since for categorical attributes the interval the data is distributed on does not affect the value of the distance. It is strongly recommended to set ```clean_bool``` to True if also ```rescale_bool``` is set to true, since the rescaling operation is heavily affected by outliers.
* ```N_SAMPLE``` : The S-RANK algorithm takes random samples of the dataset. This parameter is the number of samples to be taken. It is recommended to use at least 35.
* ```SAMPLE_SIZE``` : The number of rows to include in each sample. It is recommended to use at least 0.25% of the dataset in each sample. If possible, tale samples of at least 1% of the total dataset.

## Reference
Dash, M., & Liu, H. (2000). Feature selection for clustering. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 1805, pp. 110-121). (Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics); Vol. 1805). Springer Verlag. 

### Note:
The algorithm has been used is a small data science project and has given great results. However, i'm still testing its full capabilities.
