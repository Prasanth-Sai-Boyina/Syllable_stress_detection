from cgi import test
from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.layers import Dropout
from keras import backend as K
import keras
from sklearn import svm
from var import std, avg, test_features, test_labels, test_words

latent_dim=19
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.004)
    return z_mean + K.exp(z_log_sigma) * epsilon

model_name="vae" # it can be "sae" or "ae" also

encoder=keras.models.load_model("./"+model_name+".h5")
encoder.compile()


i=0
for v in test_features:
    test_features[i]=np.divide((v-avg), std)
    i=i+1
i=0


data_len=len(test_features)
random_array=np.zeros(data_len)
if model_name=="vae":
    svm_test=encoder([test_features, random_array])
else:
    svm_test=encoder(test_features)

clf=pickle.load(open("./"+model_name+"_classifier.sav", "rb"))
output=clf.predict(svm_test)
print(output)