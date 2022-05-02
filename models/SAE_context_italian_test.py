from tensorflow import keras
import pickle
import numpy as np
import scipy.io


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


fold= "part 2" #it can be "part 2", "part 3", "part 4", or "part 5" also




string="./context/saecdnn_italian/saecdnnitalian"+fold+".h5"
model= keras.models.load_model(string)
string="./context/saecdnn_italian/saecdnn+encoderitalian"+fold+".h5"

encoder=keras.models.load_model(string)
model.compile()
encoder.compile()

avg=np.array([ 0.13333481,  0.1446477 ,  0.11411749,  0.24227679,  0.08108932,        7.10899479, -0.41464473,  3.07106514,  0.43076186,  0.09259032,        0.08874673,  0.08810149,  0.02635661,  4.6956751 , -0.02651836,        2.20293505,  0.2642141 ,  0.43076736,  0.43073778,  0.52984166,        0.47015834,  0.48319123,  0.24348356,  0.27332521,  0.0817296 ,        0.09281364,  0.12594397,  0.14165652,  0.24019488,  0.21875761,        0.13727162,  0.01400731,  0.02180268,  0.02155907,  0.0637028 ,        0.08928136,  0.11461632,  0.88538368])
std=np.array([0.0988513 , 0.09184498, 0.0810298 , 0.15638119, 0.05784917,       1.70764467, 0.72323779, 1.50420049, 0.2054944 , 0.07367373,       0.06964175, 0.07136787, 0.02448825, 0.56165212, 0.38250157,       0.52026242, 0.16740864, 0.17844775, 0.17331636, 0.49910868,       0.49910868, 0.49971739, 0.42918448, 0.4456664 , 0.27395231,       0.2901711 , 0.33178621, 0.3486975 , 0.42720171, 0.41340382,       0.34413387, 0.11752065, 0.14603877, 0.1452387 , 0.24422276,       0.28514944, 0.31855835, 0.31855835])


test_features="to be loaded from file path"
test_labels="to be loaded from file path"
test_words="to be loaded from file path"

i=0
for v in test_features:
    test_features[i]=np.divide((v-avg), std)
    i=i+1
i=0

#svm_test=encoder([test_features, test_labels]) #for vae
svm_test=encoder(test_features) #For ae and sae

string="./classifiers/context/saecdnnitalian"+fold+".sav"
clf=pickle.load(open(string, "rb"))

pred_svm_labels= clf.predict(svm_test)
prob_labels=clf.predict_proba(svm_test)
prob_labels=np.hsplit(prob_labels, 2)[1]
post_svm_labels=make_partitions(test_words, prob_labels)
var3=calculate_accuracy(pred_svm_labels, test_labels)
print(var3)
var4=calculate_accuracy(post_svm_labels, test_labels)
print(var4)
