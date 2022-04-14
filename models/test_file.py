from tensorflow import keras

import numpy as np


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


model_name="vae" # it can be "sae" or "ae" also
language= "german" # it can be "italian" or "mixed" also
fold= "part 1" #it can be "part 2", "part 3", "part 4", or "part 5" also
type =" context" #it can be acoustic also


string="./"+type+"/"+model_name+"cdnn_"+language+"/"+model_name+"cdnn"+language+fold+".h5"
model= keras.models.load_model(string)
string="./"+type+"/"+model_name+"cdnn_"+language+"/"+model_name+"cdnn+encoder"+language+fold+".h5"
encoder=keras.models.load_model(string)
model.compile()
encoder.compile()
svm_test=encoder([test_features, test_labels])

pred_svm_labels= clf.predict(svm_test)
prob_labels=clf.predict_proba(svm_test)
prob_labels=np.hsplit(prob_labels, 2)[1]
post_svm_labels=make_partitions(test_words, prob_labels)
var3=calculate_accuracy(pred_svm_labels, test_labels)
print(var3)
var4=calculate_accuracy(post_svm_labels, test_labels)
print(var4)
 