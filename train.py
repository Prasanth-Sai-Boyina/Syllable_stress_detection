from dataloader import train_features, train_labels
from vae import vae
from ae import ae
from sae import sae
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.models import Model
from keras import losses
from keras import regularizers
from sklearn import svm

callbacks=[EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)]
"""
#For VAE training
vae.fit({"input": train_features, "lab_input":train_labels}, {"same": train_features, "stress": train_labels}
        , epochs=500, batch_size=32
        , callbacks=callbacks
        )
svm_train=encoder([train_features, train_labels])

"""

# For AE training
ae.fit({"input": train_features}, {"same": train_features, "stress": train_labels}, 
        epochs=500, batch_size=32,
        callbacks=callbacks)
svm_train=encoder(train_features)


# For SAE training
"""
sae.fit({"input": train_features}, {"same": train_features, "stress": train_labels}, 
        epochs=500, batch_size=32,
        callbacks=callbacks)
svm_train=encoder(train_features)
"""

clf= svm.SVC(probability=True)
clf.fit(svm_train, train_labels)