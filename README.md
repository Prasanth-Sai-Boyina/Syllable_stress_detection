### Syllable_stress_detection
The file Post_Part1.ipynb refers to the baseline model for mixed dataset
The file Post_Part1.ipynb refrers to the baseline model for german and italian models where dataset can be swapped
The file Post_Part2_1_VAE_pipeline.ipynb refers to the proposed model for mixed dataset
The file Post_Part2_1_VAE_pipeline_lang.ipynb refrers to the proposed model for german and italian models where dataset can be swapped

## Models Folder
This folder contains all the trained model weights.

# Directory Structure
acoustic=Models when trained with acoustic features \n
context=Models when trained with acoustic and context features \n

aecdnn_german= Normal Auto Encoder trained with german
aecdnn_italian= Normal Auto Encoder trained with italian
aecdnn_mixed= Normal Auto Encoder trained with mixed
saecdnn_german= Sparse Auto Encoder trained with german
saecdnn_italian= Sparse Auto Encoder trained with italian
saecdnn_mixed= Sparse Auto Encoder trained with mixed
vaecdnn_german= Variational Auto Encoder trained with german
vaecdnn_italian= Variational Auto Encoder trained with italian
vaecdnn_mixed= Variational Auto Encoder trained with mixed

# Filename Semantics
part i in the Filename refers to the ith fold when training the models.
If there is "encoder" in Filename, then that file refers to the trained encoder part of the specific autoencoder.
If there is no "encoder" in the Filename then that file refers to the total trained model of the specific autoencder.
