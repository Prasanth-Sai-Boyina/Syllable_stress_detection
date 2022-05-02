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


fold= "part 1" #it can be "part 2", "part 3", "part 4", or "part 5" also


string="./context/saecdnn_mixed/saecdnnmixed"+fold+".h5"
model= keras.models.load_model(string)
string="./context/saecdnn_mixed/saecdnn+encodermixed"+fold+".h5"

encoder=keras.models.load_model(string)
model.compile()
encoder.compile()

avg=np.array([ 0.12688713,  0.14006358,  0.1098526 ,  0.24107523,  0.08015861,        7.08002   , -0.42433612,  3.10882693,  0.43354382,  0.08869084,        0.08485156,  0.08515878,  0.02532137,  4.71343502, -0.0582767 ,        2.19687775,  0.2626942 ,  0.43355736,  0.4335775 ,  0.54092062,        0.45907938,  0.46915549,  0.25190286,  0.27894165,  0.08300109,        0.09546937,  0.12272563,  0.13193186,  0.23740486,  0.22500906,        0.14534252,  0.01558536,  0.02696629,  0.02500906,  0.07024284,        0.09430953,  0.12171076,  0.87828924])
std=np.array([0.09686849, 0.09092532, 0.080161  , 0.15866225, 0.05811777,       1.72903353, 0.73158156, 1.60827851, 0.20664309, 0.07102463,       0.06687988, 0.0705235 , 0.02436984, 0.55887522, 0.3777303 ,       0.52339897, 0.16566363, 0.17361942, 0.17416295, 0.49832269,       0.49832269, 0.49904771, 0.43410576, 0.44847877, 0.27588386,       0.29386216, 0.328122  , 0.33841667, 0.42549241, 0.41758829,       0.35244584, 0.12386466, 0.16198491, 0.15615252, 0.25555584,       0.29225887, 0.32695146, 0.32695146])


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

string="./classifiers/context/saecdnnmixed"+fold+".sav"
clf=pickle.load(open(string, "rb"))

pred_svm_labels= clf.predict(svm_test)
prob_labels=clf.predict_proba(svm_test)
prob_labels=np.hsplit(prob_labels, 2)[1]
post_svm_labels=make_partitions(test_words, prob_labels)
var3=calculate_accuracy(pred_svm_labels, test_labels)
print(var3)
var4=calculate_accuracy(post_svm_labels, test_labels)
print(var4)
