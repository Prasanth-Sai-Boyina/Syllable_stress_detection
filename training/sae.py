import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

original_dim = 19
intermediate_dim = 64
latent_dim = 19
ae_input = keras.Input(shape=(original_dim,), name="input")
e_features=Dense(intermediate_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(ae_input)
e_features=Dense(latent_dim, activation='relu')(e_features)
d_features=Dense(intermediate_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(e_features)
d_pred=Dense(original_dim, activation='sigmoid' ,name='same')(d_features)
encoder=keras.Model(inputs=[ae_input],
                    outputs=[e_features]
                    )
clf_features=Dense(128, activation='relu')(e_features)
clf_features=Dropout(0.45)(clf_features)
clf_features=Dense(64, activation='relu')(clf_features)
clf_features=Dense(32, activation='relu')(clf_features)
clf_features=Dense(16, activation='relu')(clf_features)
clf_features=Dense(8, activation='relu')(clf_features)
clf_features=Dense(4, activation='relu')(clf_features)
clf_features=Dense(2, activation='relu')(clf_features)
clf_pred=Dense(1, activation='sigmoid', name='stress')(clf_features)
sae = keras.Model(
    inputs=[ae_input],
    outputs=[d_pred, clf_pred],
)
sae.compile(
    optimizer='adam',
    loss={
        "same":keras.losses.MeanSquaredError(),
        "stress":keras.losses.BinaryCrossentropy(),
    },
    loss_weights={"same": 0.85, "stress":1},
)
