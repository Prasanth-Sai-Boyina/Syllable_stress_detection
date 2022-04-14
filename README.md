# Syllable_stress_detection
The file Post_Part1.ipynb refers to the baseline model for mixed dataset <br />
The file Post_Part1.ipynb refrers to the baseline model for german and italian models where dataset can be swapped <br />
The file Post_Part2_1_VAE_pipeline.ipynb refers to the proposed model for mixed dataset <br />
The file Post_Part2_1_VAE_pipeline_lang.ipynb refrers to the proposed model for german and italian models where dataset can be swapped <br />

## Models Folder
This folder contains all the trained model weights.

### Directory Structure
acoustic=Models when trained with acoustic features <br />
context=Models when trained with acoustic and context features <br /> <br />

aecdnn_german= Normal Auto Encoder trained with german <br />
aecdnn_italian= Normal Auto Encoder trained with italian <br />
aecdnn_mixed= Normal Auto Encoder trained with mixed <br />
saecdnn_german= Sparse Auto Encoder trained with german <br />
saecdnn_italian= Sparse Auto Encoder trained with italian <br />
saecdnn_mixed= Sparse Auto Encoder trained with mixed <br />
vaecdnn_german= Variational Auto Encoder trained with german <br />
vaecdnn_italian= Variational Auto Encoder trained with italian <br />
vaecdnn_mixed= Variational Auto Encoder trained with mixed <br />

### Filename Semantics
part i in the Filename refers to the ith fold when training the models. <br />
If there is "encoder" in Filename, then that file refers to the trained encoder part of the specific autoencoder. <br />
If there is no "encoder" in the Filename then that file refers to the total trained model of the specific autoencder. <br />
