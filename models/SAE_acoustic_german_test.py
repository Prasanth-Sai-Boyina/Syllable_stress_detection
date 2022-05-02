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


fold= "part 4" #it can be "part 2", "part 3", "part 4", or "part 5" also




string="./acoustic/saecdnn_german/saecdnngerman"+fold+".h5"
model= keras.models.load_model(string)
string="./acoustic/saecdnn_german/saecdnn+encodergerman"+fold+".h5"

encoder=keras.models.load_model(string)
model.compile()
encoder.compile()

avg=np.array([ 0.12442026,  0.14095997,  0.10932499,  0.25408744,  0.08335438,        6.93530121, -0.40927128,  3.16525484,  0.44114357,  0.08496958,        0.08074326,  0.08238804,  0.0240726 ,  4.75486395, -0.11426323,        2.17045095,  0.25264484,  0.44114593,  0.44113643])
std=np.array([0.09391503, 0.08989826, 0.07856982, 0.16715627, 0.06055207,      1.6512075 , 0.74861334, 1.71107923, 0.20551775, 0.06497229,       0.06054434, 0.06484704, 0.02228486, 0.52661585, 0.35664232,       0.48094181, 0.15164808, 0.16857016, 0.18336748])


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

string="./classifiers/acoustic/saecdnngerman"+fold+".sav"
clf=pickle.load(open(string, "rb"))

pred_svm_labels= clf.predict(svm_test)
prob_labels=clf.predict_proba(svm_test)
prob_labels=np.hsplit(prob_labels, 2)[1]
post_svm_labels=make_partitions(test_words, prob_labels)
var3=calculate_accuracy(pred_svm_labels, test_labels)
print(var3)
var4=calculate_accuracy(post_svm_labels, test_labels)
print(var4)