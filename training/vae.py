import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import keras

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.004)
    return z_mean + K.exp(z_log_sigma) * epsilon
 

original_dim = 19
intermediate_dim = 64
latent_dim = 19

#Vae
lab_input = keras.Input(shape=(1,), name="lab_input")
ae_input = keras.Input(shape=(original_dim,), name="input")
h = layers.Dense(intermediate_dim, activation='relu')(ae_input)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

z = layers.Lambda(sampling)([z_mean, z_log_sigma])
encoder = keras.Model(inputs=[ae_input, lab_input], 
                      outputs=z
                      )
x = Dense(latent_dim, activation='relu')(z)
x = Dense(intermediate_dim, activation='relu')(x)
ae_output = Dense(original_dim, activation='sigmoid', name="same")(x)

#classifier
clf_features=Dense(128, activation='relu')(z)
clf_features=Dropout(0.45)(clf_features)
clf_features=Dense(64, activation='relu')(clf_features)
clf_features=Dense(32, activation='relu')(clf_features)
clf_features=Dense(16, activation='relu')(clf_features)
clf_features=Dense(8, activation='relu')(clf_features)
clf_features=Dense(4, activation='relu')(clf_features)
clf_features=Dense(2, activation='relu')(clf_features)
clf_pred=Dense(1, activation='sigmoid', name='stress')(clf_features)


reconstruction_loss = keras.losses.mean_squared_error(ae_input, ae_output)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
l1=keras.losses.binary_crossentropy(lab_input ,clf_pred)
clf_loss = K.mean(l1)


vae = keras.Model(
    inputs=[ae_input, lab_input],
    outputs=[ae_output, clf_pred],
)
vae.add_loss({"same":vae_loss})
vae.add_loss({"stress":clf_loss})

vae.compile(
    optimizer='adam',
    loss_weights={"same": 0.3, "stress":1},
)


