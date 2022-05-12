#loading data
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import scipy.io
GER_tr = scipy.io.loadmat('/content/drive/MyDrive/postprocessing/GER_train.mat');

GER_train = GER_tr['GER_train']
GER_ytrain = GER_train[0]
GER_wtrain =GER_train[1]
GER_xtrain = GER_train[2:].T


train_features = GER_xtrain
train_labels = GER_ytrain
train_words=GER_wtrain

train_size=train_features.shape[0]


avg_trainfeat=np.mean(train_features, axis=0)
std_trainfeat=np.std(train_features, axis=0)

#Normalization
i=0
for v in train_features:
    train_features[i]=np.divide((v-avg_trainfeat), std_trainfeat)
    i=i+1
i=0

