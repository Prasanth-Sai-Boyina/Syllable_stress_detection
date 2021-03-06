import numpy as np
import scipy.io
GER_tr = scipy.io.loadmat('./data/train.mat')
GER_te = scipy.io.loadmat('./data/test.mat')

GER_train = GER_tr['GER_train']
GER_ytrain = GER_train[0]
GER_wtrain =GER_train[1]
GER_xtrain = GER_train[2:].T


GER_test = GER_te['GER_test']
GER_ytest = GER_test[0]
GER_wtest =GER_test[1]
GER_xtest = GER_test[2:].T


train_features = GER_xtrain
train_labels = GER_ytrain
train_words=GER_wtrain
train_size=train_features.shape[0]

test_features = GER_xtest
test_labels = GER_ytest
test_words=GER_wtest

avg_trainfeat=np.mean(train_features, axis=0)
std_trainfeat=np.std(train_features, axis=0)

avg=np.array([ 0.12688713,  0.14006358,  0.1098526 ,  0.24107523,  0.08015861,        7.08002   , -0.42433612,  3.10882693,  0.43354382,  0.08869084,        0.08485156,  0.08515878,  0.02532137,  4.71343502, -0.0582767 ,        2.19687775,  0.2626942 ,  0.43355736,  0.4335775 ])
std=np.array([0.09686849, 0.09092532, 0.080161  , 0.15866225, 0.05811777,       1.72903353, 0.73158156, 1.60827851, 0.20664309, 0.07102463,       0.06687988, 0.0705235 , 0.02436984, 0.55887522, 0.3777303 ,       0.52339897, 0.16566363, 0.17361942, 0.17416295])

#Normalization
i=0
for v in train_features:
    train_features[i]=np.divide((v-avg_trainfeat), std_trainfeat)
    i=i+1
i=0

