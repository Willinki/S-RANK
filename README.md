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
  from FeatureRanker import SRANK
  ```
Then, in order to use the algorithm:
  ```
  ranker = SRANK()
  ranker.apply(df_big, vars_type, discrete_var_list, clean_bool = False, rescale_bool = False, 
               N_SAMPLE = 40, SAMPLE_SIZE = 50)
  ```
  
Arguments:
* ```df_big```: the whole data. It has to be a pandas.Dataframe object. Please make sure that all variable dtypes are inferred correctly. It is important to have numeric variables and non-numeric variables (dtype: object) inferred correctly. Up to now, the algorithm is able to 
* ```vars_type``` the type of data in the dataframe. String that can take 3 values: ```continous``` if all the variables have continous numerical values. ```discrete``` if all the variables are categorical (encoded as string or number makes no difference), ```mixed``` if there's both.

to be continued...
