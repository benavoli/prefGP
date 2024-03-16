# INITIALIZE TOOLS
import sys
import numpy as np
import pandas as pd
from math import ceil
from subprocess import call
from itertools import islice
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, dok_matrix
import os
import itertools
import pickle

from sklearn.model_selection import train_test_split


def create_matrix(data, users_col, items_col, ratings_col, threshold = None):
    """
    creates the sparse user-item interaction matrix,
    if the data is not in the format where the interaction only
    contains the positive items (indicated by 1), then use the 
    threshold parameter to determine which items are considered positive
    
    Parameters
    ----------
    data : DataFrame
        implicit rating data

    users_col : str
        user column name

    items_col : str
        item column name
    
    ratings_col : str
        implicit rating column name

    threshold : int, default None
        threshold to determine whether the user-item pair is 
        a positive feedback

    Returns
    -------
    ratings : scipy sparse csr_matrix, shape [n_users, n_items]
        user/item ratings matrix

    data : DataFrame
        implict rating data that retains only the positive feedback
        (if specified to do so)
    """
    if threshold is not None:
        indx = np.where(data[ratings_col] >= threshold)[0]
        indm=np.unique(data.item_id[indx])
        data=data.iloc[indx,:]
        data[ratings_col] = 1

    # this ensures each user has at least 2 records to construct a valid
    # train and test split in downstream process, note we might purge
    # some users completely during this process
    data_user_num_items = (data
                         .groupby('user_id')
                         .agg(**{'num_items': ('item_id', 'count')})
                         .reset_index())
    data = data.merge(data_user_num_items, on='user_id', how='inner')
    data = data[data['num_items'] > 50]
    #print(data)
    
    for col in (items_col, users_col, ratings_col):
        data[col] = data[col].astype('category')

    ratings = csr_matrix((data[ratings_col],
                          (data[users_col].cat.codes, data[items_col].cat.codes)))
    ratings.eliminate_zeros()
    return ratings, data, indm

def create_train_test(ratings, test_size = 0.2, seed = 1234):
    """
    split the user-item interactions matrix into train and test set
    by removing some of the interactions from every user and pretend
    that we never seen them
    
    Parameters
    ----------
    ratings : scipy sparse csr_matrix, shape [n_users, n_items]
        The user-item interactions matrix
    
    test_size : float between 0.0 and 1.0, default 0.2
        Proportion of the user-item interactions for each user
        in the dataset to move to the test set; e.g. if set to 0.2
        and a user has 10 interactions, then 2 will be moved to the
        test set
    
    seed : int, default 1234
        Seed for reproducible random splitting the 
        data into train/test set
    
    Returns
    ------- 
    train : scipy sparse csr_matrix, shape [n_users, n_items]
        Training set
    
    test : scipy sparse csr_matrix, shape [n_users, n_items]
        Test set
    """
    assert test_size < 1.0 and test_size > 0.0

    # Dictionary Of Keys based sparse matrix is more efficient
    # for constructing sparse matrices incrementally compared with csr_matrix
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)
    
    # for all the users assign randomly chosen interactions
    # to the test and assign those interactions to zero in the training;
    # when computing the interactions to go into the test set, 
    # remember to round up the numbers (e.g. a user has 4 ratings, if the
    # test_size is 0.2, then 0.8 ratings will go to test, thus we need to
    # round up to ensure the test set gets at least 1 rating)
    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = rstate.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0
    
    train, test = train.tocsr(), test.tocsr()
    return train, test
    

def make_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        if arr[i] > 0:
            for j in range(len(arr)):
                if arr[j] == 0 and i != j:
                    pairs.append((i, j))
    return pairs

def make_CA_RA(arr):
    CA=[]
    RA=[]
    ind_RA=np.where(arr==0)[0].tolist()
    for i in range(len(arr)):
        if arr[i] > 0:
            CA.append([i])
            RA.append(ind_RA)
    return CA,RA


def generateMovieChoiceData(nmovies,nusers,dir_to_save,data_dir='./ml-100k'): 
    u_data = os.path.join(data_dir, 'u.data')
    u_genre = os.path.join(data_dir, 'u.genre')
    u_item = os.path.join(data_dir, 'u.item')
    u_pers = os.path.join(data_dir, 'u.user')

    # we will not be using the timestamp column
    print("Reading data from folder",data_dir,"...")
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df_data = pd.read_csv(u_data, sep = '\t', names = names)
    print('data dimension: \n', df_data.shape)

    df_genre = pd.read_csv(u_genre, sep = '|',header=None)
    cols=df_genre.loc[:,0].values.tolist()

    df_item= pd.read_csv(u_item, sep = '|',header=None, encoding = "ISO-8859-1",index_col=0)
    df_item=df_item.drop(columns=[3,4])
    df_item.columns=["title","date"]+cols
    df_item['date']=df_item['date'].apply(pd.to_datetime).dt.year.astype(float)

    df_user_personal= pd.read_csv(u_pers, sep = '|',header=None, encoding = "ISO-8859-1",index_col=0)

    # DATA PREPROCESSING

    # users' features scaling
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    ct = ColumnTransformer(
        [("norm1", StandardScaler(), [0]),
        ("norm2", OneHotEncoder(), [1,2])    
        ])

    items_col = 'item_id'
    users_col = 'user_id'
    ratings_col = 'rating'
    threshold = 3# we only consider movies that received more than 3 stars rating

    #scale users features
    Xuser = np.array(ct.fit_transform(df_user_personal.iloc[:,0:-1]).todense())

    # Make user movie interaction matrix
    X, df_user, _ = create_matrix(df_data, users_col, items_col, ratings_col, threshold)
    X = X.toarray()

    #filter user that were not considered in X
    indu=np.unique(df_user['user_id'])-1
    Xuser = Xuser[indu,:]


    # Select a subset of movies and users
    # If we do not pass a nmovies/nusers we keep all possible combinations
    if nmovies is None:
        nmovies = X.shape[1]
    if nusers is None:
        nusers = X.shape[0]   

    X = X[:,0:nmovies]

    print("number of movies selected: ", nmovies)


    #make all possibles preference pairs for i-th user 
    Allchoices=[]
    indgood=[]
    for i in range(nusers): 
        ind1=np.where(X[i,:]==1)[0]
        ind0=np.where(X[i,:]==0)[0]
        if (len(ind1)>0)&(len(ind0))>0:
            indgood.append(i)
            c=np.vstack(list(itertools.product(ind1,ind0)))
            Allchoices.append(c)
            
    #filter out uses that have watched all movies or none 
    Xuser=Xuser[indgood,:]
    X=X[indgood,:]
    print("Size of users/movies matrix ", X.shape)
    print("Size of user features ",Xuser.shape)

    if len(Allchoices)!=Xuser.shape[0]:
        raise ValueError("Error")
    
    # augment data to represent label-preferences as object-preference
    largeX=np.tile(Xuser.T,X.shape[1]).T
    # add column which acts as movies indicator and as an additional features
    colind = np.tile(np.arange(X.shape[1])[:,None],Xuser.shape[0]).flatten()
    #join the two
    largeX = np.hstack([largeX,colind[:,None]])
    print("Size of tiled input feature vector:",largeX.shape)

    #redefine the choices indexes with respect to the rows of largeX:
    nu=len(Allchoices)
    Allchoices1=[]
    for i in range(len(Allchoices)):
        Allchoices1.append(
            Allchoices[i]*nu+i
            )
    Allchoices1v=np.vstack(Allchoices1)

    print("Total number of choices: ", len(Allchoices1v))

    if dir_to_save is not None:
        with open(dir_to_save+"/largeX.pkl", 'wb') as handle:
            pickle.dump(largeX, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_to_save+"/allChoices.pkl", 'wb') as handle:
            pickle.dump(Allchoices1v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return largeX, Allchoices1v