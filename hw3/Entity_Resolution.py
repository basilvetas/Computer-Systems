
# coding: utf-8

# ### Package Installation Instructions

# In[1]:


# pip3 install numpy
# pip3 install pandas
# pip3 install recordlinkage
# https://www.scipy.org/install.html
# pip3 install -U scikit-learn
# https://matplotlib.org/users/installing.html
# pip3 install missingno


# ### Base Imports

# In[2]:


# for data analysis 
import numpy as np
import pandas as pd
import recordlinkage

# for plotting missing data
import matplotlib.pyplot as plt
import missingno as msno


# ### Set Options

# In[3]:


# display options
# %matplotlib inline
# pd.options.display.max_rows = 999
# pd.options.display.max_columns = 999


# In[4]:


# Enable multiple cell outputs
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# ### Preprocessing

# In[5]:


from recordlinkage.preprocessing import clean, phonenumbers

def preprocess(df1, df2):
        
    # set index to the id column
    df1 = df1.set_index('id')
    df2 = df2.set_index('id')
    
    # replace empty cells with NaN
    df1 = df1.replace("", np.nan)
    df2 = df2.replace("", np.nan)
        
    # drop country, locality and region
    df1 = df1.drop(['country', 'locality', 'region'], axis=1)
    df2 = df2.drop(['country', 'locality', 'region'], axis=1)

    # remove all non-numbers from phone & convert to numeric
    df1.loc[:, 'phone'] = pd.to_numeric(phonenumbers(df1.loc[:, 'phone']))
    df2.loc[:, 'phone'] = pd.to_numeric(phonenumbers(df2.loc[:, 'phone']))
    
    # convert postal_code to numeric
    df1.loc[:, 'postal_code'] = pd.to_numeric(df1.loc[:, 'postal_code'])
    df2.loc[:, 'postal_code'] = pd.to_numeric(df2.loc[:, 'postal_code'])
    
    # clean street_address & website
    df1.loc[:, 'street_address'] = clean(df1.loc[:, 'street_address'])
    df1.loc[:, 'website'] = clean(df1.loc[:, 'website'])
    
    df2.loc[:, 'street_address'] = clean(df2.loc[:, 'street_address'])    
    df2.loc[:, 'website'] = clean(df2.loc[:, 'website'])
    
    # convert NaNs to 0s for numerics
    df1.loc[:,['latitude', 'longitude', 'phone', 'postal_code']] =     df1.loc[:,['latitude', 'longitude', 'phone', 'postal_code']].replace(np.nan, 0)
    
    df2.loc[:,['latitude', 'longitude', 'phone', 'postal_code']] =     df2.loc[:,['latitude', 'longitude', 'phone', 'postal_code']].replace(np.nan, 0)

    return df1, df2


# ### Preprocess Matches

# In[6]:


def preprocess_matches(matches):
    
    # set multiindex to locu_id and foursquare_id
    matches = matches.set_index(['locu_id', 'foursquare_id'])
    
    # drop matches I disagree with
    matches = matches.drop('c170270283ef870d546b', level='locu_id') # foursquare_id: 51eb7eed498e401ec51196b6
    matches = matches.drop('496bd5b462f08383d880', level='locu_id') # foursquare_id: 3fd66200f964a5209eea1ee3
    matches = matches.drop('9ea3254360d0fe59177e', level='locu_id') # foursquare_id: 4dc597c57d8b14fb462ed076
    matches = matches.drop('edeba23f215dcc702220', level='locu_id') # foursquare_id: 51a11cbc498e4083823909f1

    # create a dataframe for both fourquare and locu of pairs that get matched
#     tuples = list(matches.index)
#     locu_index = [i[0] for i in tuples]
#     four_index = [i[1] for i in tuples]
#     locu_matches = locu_train.loc[locu_index]
#     four_matches = four_train.loc[four_index]

    # for viewing full match dataset
#     temp = matches.reset_index().join(four_matches,on=['foursquare_id'])
#     match_pairs = temp.join(locu_matches,on=['locu_id'],lsuffix='_foursquare',rsuffix='_locu').set_index(matches.index.names)
    
#     cols = np.array(match_pairs.columns.tolist())
#     order = [0,7,1,8,2,9,3,10,4,11,5,12,6,13]
#     cols = list(cols[order])
#     matches_reordered = match_pairs[cols]
    
    return matches


# ### Index Pairs

# In[7]:


from recordlinkage.base import BaseIndexator

def index_pairs(df1, df2):
    indexer = recordlinkage.FullIndex() # BlockIndex(on='postal_code')
    return indexer.index(df1, df2)

    # Customer Indexer
    # class FirstLetterOfNameIndex(BaseIndexator):
    #     """Custom class for indexing"""

    #     def __init__(self, letter):
    #         super(FirstLetterOfNameIndex, self).__init__()

    #         # the letter to save
    #         self.letter = letter

    #     def _link_index(self, df_a, df_b):
    #         """Make record pairs that agree on the first letter of the given name."""

    #         # Select records with names starting with a 'letter'.
    #         a_startswith_w = df_a[df_a['name'].str.startswith(self.letter) == True]
    #         b_startswith_w = df_b[df_b['name'].str.startswith(self.letter) == True]

    #         # Make a product of the two numpy arrays
    #         return pd.MultiIndex.from_product(
    #             [a_startswith_w.index.values, b_startswith_w.index.values],
    #             names=[df_a.index.name, df_b.index.name]
    #         )

    # for letter in 'abcdefghijklmnopqrstuvwxyz':
    #     indexer = FirstLetterOfNameIndex(letter)
    #     candidate_pairs = candidate_pairs | indexer.index(locu_train, four_train)


# ### Compare Strings

# In[8]:


def compare_strings(df1, df2, cand_pairs):

    compare = recordlinkage.Compare()

    # initialise similarity measurement algorithms
    # compare.string('country', 'country', method='levenshtein', label='country')
    # compare.string('locality', 'locality', method='levenshtein', label='locality')
    compare.geo('latitude', 'longitude', 'latitude', 'longitude', scale=1, label='geo_coord')
    compare.string('name', 'name', method='levenshtein', label='name')
    compare.numeric('phone', 'phone', scale=1, label='phone')
    compare.numeric('postal_code', 'postal_code', scale=1, label='postal_code')
    # compare.string('region', 'region', method='levenshtein', label='region')
    compare.string('street_address', 'street_address', method='levenshtein', label='street_address')
    compare.string('website', 'website', method='levenshtein', label='website')

    # compute similarity measurements
    return compare.compute(cand_pairs, df1, df2)


# ### Train Test Split

# In[9]:



# parameters: 
#   x_all, a data_frame of all your comparison vectors
#   y_all_matches, a data_frame with 0 columns and multiindexed with the matching pairs
# returns:
#   x_train, a subset of x_all that will be used for model training
#   y_train_matches_index, a multiindex object of the the matching pairs
def traintestsplit(x_all, y_all_matches):
    
    tuples = list(y_all_matches.index)
    y_matches_index = pd.MultiIndex.from_tuples(tuples, names=['locu_id', 'foursquare_id'])
    
    x_train = x_all.sample(frac=.9, random_state=158)
    y_train_matches_index = x_train.index & y_matches_index
#     x_train_features = x_train.loc[y_matches_index]
#     x_train_features
    
    return x_train, y_train_matches_index, y_matches_index


# ### Predict

# In[10]:


### 
#Predict the match status for all record pairs
# parameters:
#   x, a data_frame of all your comparison vectors
# returns:
#   results_index, a multiindex object of predicted matches
###
def predict(x, model):    
    results_index = model.predict(x).set_names(['locu_id', 'foursquare_id'])
    return results_index


# ### Get Matches

# In[11]:


def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    four_train = pd.read_json(foursquare_train_path)
    locu_train = pd.read_json(locu_train_path) 

    four_test = pd.read_json(foursquare_test_path)
    locu_test = pd.read_json(locu_test_path)

    matches_train = pd.read_csv(matches_train_path)
    
    # visualize missing data
#     msno.matrix(four_train)
#     msno.matrix(locu_train)
#     msno.matrix(four_test)
#     msno.matrix(locu_test)
    
    locu_train, four_train = preprocess(locu_train, four_train)
    locu_test, four_test = preprocess(locu_test, four_test)
    matches_train = preprocess_matches(matches_train)
    
    candidate_pairs = index_pairs(locu_train, four_train)
    test_candidate_pairs = index_pairs(locu_test, four_test)
#     print (len(locu_train), len(four_train), len(candidate_pairs))
#     print (len(locu_test), len(four_test), len(test_candidate_pairs))
    
    features = compare_strings(locu_train, four_train, candidate_pairs)
    test_features = compare_strings(locu_test, four_test, test_candidate_pairs)
    
#     features = features.loc[features['street_address'] > .1]
#     features = features.loc[features['name'] > .1]

    train_pairs, train_matches_index, all_matches_index = traintestsplit(features, matches_train)
    
    # Train Logistic Regression classifier
    logreg = recordlinkage.LogisticRegressionClassifier()
    logreg.learn(train_pairs, train_matches_index)
#     print ("LogReg Intercept: ", logreg.intercept)
#     print ("LogReg Coefficients: ", logreg.coefficients)

    # Train SVM classifier
    svm = recordlinkage.SVMClassifier()
    svm.learn(train_pairs, train_matches_index)
    
    # Predict on training data with both classifiers
    svm_results_index = predict(features, svm)
    logreg_results_index = predict(features, logreg)
    
    # To view pairs
#     features.index = features.index.rename(['locu_id', 'foursquare_id'])
#     train_matches = features.loc[svm_results_index]
#     train_matches
    
    # Training results     
    svm_confn_matrix = recordlinkage.confusion_matrix(all_matches_index, svm_results_index, len(features))
#     print("SVM Confusion Matrix: ", svm_confn_matrix)
#     print("SVM Precision: ", recordlinkage.precision(svm_confn_matrix))
#     print("SVM Recall:    ", recordlinkage.recall(svm_confn_matrix))
#     print("SVM Accuracy:  ", recordlinkage.accuracy(svm_confn_matrix))
#     print("SVM F1 Score:  ", recordlinkage.fscore(svm_confn_matrix))
    
    logreg_confn_matrix = recordlinkage.confusion_matrix(all_matches_index, logreg_results_index, len(features))
#     print("Logistic Regression Confusion Matrix: ", logreg_confn_matrix)
#     print("Logistic Regression Precision: ", recordlinkage.precision(logreg_confn_matrix))
#     print("Logistic Regression Recall:    ", recordlinkage.recall(logreg_confn_matrix))
#     print("Logistic Regression Accuracy:  ", recordlinkage.accuracy(logreg_confn_matrix))
#     print("Logistic Regression F1 Score:  ", recordlinkage.fscore(logreg_confn_matrix))
    
    # Predict on test data with SVM
    test_results_index = predict(test_features, svm)
    
    # Format and write to CSV    
    test_features.index = test_features.index.rename(['locu_id', 'foursquare_id'])
    test_match_pairs = test_features.loc[test_results_index]
    matches_test = test_match_pairs.drop(test_match_pairs.columns[::], axis=1)        
#     matches_test
    matches_test.to_csv('matches_test.csv')
    
    # create a dataframe for both fourquare and locu of pairs that get matched
    test_tuples = list(matches_test.index)
    test_locu_index = [i[0] for i in test_tuples]
    test_four_index = [i[1] for i in test_tuples]
    test_locu_matches = locu_test.loc[test_locu_index]
    test_four_matches = four_test.loc[test_four_index]

    # for viewing full match dataset
    temp = matches_test.reset_index().join(test_four_matches,on=['foursquare_id'])
    test_match_pairs = temp.join(test_locu_matches,on=['locu_id'],lsuffix='_foursquare',rsuffix='_locu').set_index(matches_test.index.names)

    cols = np.array(test_match_pairs.columns.tolist())
    order = [0,7,1,8,2,9,3,10,4,11,5,12,6,13]
    cols = list(cols[order])
    test_matches_reordered = test_match_pairs[cols]
#     display(test_matches_reordered)    
#     print("Successfully wrote results to matches_test.csv")
    return
        
get_matches("locu_train_hard.json", "foursquare_train_hard.json", "matches_train_hard.csv", "locu_test_hard.json", "foursquare_test_hard.json")

