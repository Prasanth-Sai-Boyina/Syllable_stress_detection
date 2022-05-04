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

def calculate_accuracy(arr1, arr2):
  count=0
  for itr1, itr2 in zip(arr1, arr2):
    if itr1==itr2:
      count+=1
  return count/len(arr1)

def make_partitions(arr_words, arr_labels):
  v=[]
  np.array(v)
  temp=[]
  for i in range(len(arr_words)-1):
    word=arr_words[i]
    next_word=arr_words[i+1]
    temp.append(arr_labels[i][0])
    if word!=next_word:
      numpy_temp=np.array(temp)
      temp_max=np.amax(numpy_temp)
      numpy_temp=np.divide(numpy_temp, temp_max)
      v=np.concatenate((v, numpy_temp), axis=None)
      temp.clear()
    if (i==len(arr_words)-2):
      temp.append(arr_labels[i+1][0])
      numpy_temp=np.array(temp)
      temp_max=np.amax(numpy_temp)
      numpy_temp=np.divide(numpy_temp, temp_max)
      v=np.concatenate((v, numpy_temp), axis=None)
      temp.clear()
  v1=[]
  for i in v:
    if i==1:
      v1.append(1)
    else:
      v1.append(0)
  return v1

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.004)
    return z_mean + K.exp(z_log_sigma) * epsilon

model_name="vae" # it can be "sae" or "ae" also
language= "german" # it can be "italian" or "mixed" also
fold= "part 1" #it can be "part 2", "part 3", "part 4", or "part 5" also
type ="context" #it can be "acoustic" also

if type=="context":
  latent_dim=38
else:
  latent_dim=19

string="./"+type+"/"+model_name+"cdnn_"+language+"/"+model_name+"cdnn"+language+fold+".h5"
model= keras.models.load_model(string)
string="./"+type+"/"+model_name+"cdnn_"+language+"/"+model_name+"cdnn+encoder"+language+fold+".h5"
encoder=keras.models.load_model(string)
model.compile()
encoder.compile()

avg="to be loaded"
std="to be loaded"

test_features="to be loaded from file path"
test_labels="to be loaded from file path"
test_words="to be loaded from file path"

i=0
for v in test_features:
    test_features[i]=np.divide((v-avg), std)
    i=i+1
i=0



svm_test=encoder([test_features, test_labels]) #for vae
#svm_test=encoder(test_features) #For ae and sae

string="./classifiers/"+type+"/"+model_name+"cdnn"+language+fold+".sav"
clf=pickle.load(open(string, "rb"))

pred_svm_labels= clf.predict(svm_test)
prob_labels=clf.predict_proba(svm_test)
prob_labels=np.hsplit(prob_labels, 2)[1]
post_svm_labels=make_partitions(test_words, prob_labels)
var3=calculate_accuracy(pred_svm_labels, test_labels)
print(var3)
var4=calculate_accuracy(post_svm_labels, test_labels)
print(var4)
