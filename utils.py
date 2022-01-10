import pandas as pd
import numpy as np
import random

'''
    BATCH GENERATORS
'''

def batch_generator(df, batch_size):
    '''
        randomized generator returing N = batch_size samples at a time
        respecting global class proprion in each batch
        (batch_size, max_tiles_per_batch, feature_size)
        padding with zeros for constant n_tiles across samples        
    '''
    df_neg = df[df.Target==0]
    df_pos = df[df.Target==1]
    
    n_neg_samples = int(batch_size * len(df_neg)/len(df))
    n_pos_samples = batch_size - n_neg_samples
    
    while True:
        # randomly sample N*ratio negative samples and N*(1-ratio) positive samples
        df_batch_neg = df_neg.sample(n=n_neg_samples)
        df_batch_pos = df_pos.sample(n=n_pos_samples)
        df_batch = pd.concat([df_batch_neg,df_batch_pos])
        # shuffle
        df_batch = df_batch.sample(frac=1)
        # get labels in batch
        Y = df_batch.Target.values
        # compute the maximum number of tiles contained by an image in  the current batch
        n_tiles_per_batch = df_batch['tiles_count'].max()
        # get the features in batch
        X = []
        for _, row in df_batch.iterrows(): 
            # get filenames
            filename = row['Path']
            # get features from file
            features = np.load(filename)[:, 3:]
            # get number of tiles in current samples
            n_tiles_per_image = features.shape[0]
            # pad with zeros if neccesary
            if n_tiles_per_image < n_tiles_per_batch:
                features = np.pad(features, ((n_tiles_per_batch-n_tiles_per_image, 0), (0, 0)))
            X.append(features)  
        X = np.stack(X, axis=0)
        yield X, Y
        
        
def naive_batch_generator(df, batch_size):
    '''
        randomized generator returing N = batch_size samples at a time
        (batch_size, max_tiles_per_batch, feature_size)
        padding with zeros for constant n_tiles across samples        
    '''
    list_of_features = [np.load(filename)[:, 3:] for filename in df['Path']]
    tiles_per_image = df['tiles_count'].values
    labels = df['Target'].values

    while True:
        # randomly sample N indeces
        idxs = random.sample(range(0, len(df)), batch_size)
        # get the maximum number of tiles contained by an image in the current batch
        n_tiles_per_batch = tiles_per_image[idxs].max()
        # get the features in batch
        list_of_features_per_batch = [list_of_features[i] for i in idxs]
        X = []
        # pad with zeros to compensate for missing tiles
        for features in list_of_features_per_batch: 
            n_tiles_per_image = features.shape[0]
            # pad with zeros if neccesary
            if n_tiles_per_image < n_tiles_per_batch:
                features = np.pad(features, ((n_tiles_per_batch-n_tiles_per_image, 0), (0, 0)))
            X.append(features)            
        X = np.stack(X, axis=0)
        yield X, labels[idxs]
        
        
def test_batch_generator(df):
    '''
        ordered generator returing one sample at a time
        (1, n_tiles_per_sample, n_features)
    '''
    list_of_features = [np.load(filename)[:, 3:] for filename in df['Path']]
    n_batches = len(df)
    batch_id = 0

    while True:  
        batch_id = batch_id + 1 if batch_id < n_batches-1 else 0
        yield list_of_features[batch_id][None,...]
       
    
'''
    DATA HELPERS
'''

def get_number_tiles(filenames):
    n_tiles_per_image = []
    for f in filenames:
        patient_features = np.load(f)
        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]
        n_tiles_per_image.append(patient_features.shape[0])
    return n_tiles_per_image
