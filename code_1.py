# Importing all the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import concatenate
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.manifold import TSNE

# Preprocessing the data after loading it

RNA=pd.read_csv('file_path',sep='\t')
RNA

Proteomics = pd.read_csv('file_path',sep='\t')
Proteomics

RNA=RNA.T
Proteomics=Proteomics.T

# Dropping the unwanted columns
RNA.drop('sample',axis=0,inplace=True)

# Replace nan with 0
RNA=RNA.replace(np.nan,0)

Proteomics.drop('sample_id',axis=0,inplace=True)
Proteomics=Proteomics.resample(np.nan,0)

RNA.reset_index(drop=True,inplace=True)
Proteomics.reset_index(drop=True,inplace=True)


# Selecting the features and labels to feed the model that we build
X_RNA = RNA.iloc[:,0:(RNA.shape[1]-1)]
Y_RNA = RNA.iloc[:,RNA.shape[1]-1]
X_Proteomics = Proteomics.iloc[:,0:(Proteomics.shape[1]-1)]
Y_Proteomics = Proteomics.iloc[:,Proteomics.shape[1]-1]


# The model
ncol_RNA = X_RNA.shape[1]
# the input layer
input_dim_RNA = Input(shape = (ncol_RNA,), name = "RNA")
ncol_Proteomics = X_Proteomics.shape[1]
input_dim_Proteomics = Input(shape = (ncol_Proteomics,), name = "Proteomics")

encoding_dim_RNA = 50
encoding_dim_Proteomics = 10

# the dense layers
encoded_RNA = Dense(encoding_dim_RNA, activation = 'linear',
                         name = "Encoder_RNA")(input_dim_RNA)
encoded_Proteomics = Dense(encoding_dim_Proteomics, activation = 'linear',
                             name = "Encoder_Proteomics")(input_dim_Proteomics)

# Concatenating the two inputs that we got
merge = concatenate([encoded_RNA, encoded_Proteomics])

# Bottleneck layer formed where the major and required features are present
bottleneck = Dense(50, kernel_initializer = 'uniform', activation = 'linear',
                   name = "Bottleneck")(merge)

# Regenerating the inputs from the bottleneck features
merge_inverse = Dense(encoding_dim_RNA + encoding_dim_Proteomics,
                      activation = 'elu', name = "Concatenate_Inverse")(bottleneck)
decoded_RNA= Dense(ncol_RNA, activation = 'sigmoid',
                         name = "Decoder_RNA")(merge_inverse)
decoded_Proteomics = Dense(ncol_Proteomics, activation = 'sigmoid',
                             name = "Decoder_Proteomics")(merge_inverse)

autoencoder = Model(inputs = [input_dim_RNA, input_dim_Proteomics],
                    outputs = [decoded_RNA, decoded_Proteomics])

# Compile step
autoencoder.compile(optimizer = 'adam',
                    loss={'Decoder_RNA': 'mean_squared_error',
                          'Decoder_Proteomics': 'mean_squared_error'},metrics=['accuracy'])
autoencoder.summary()

plot_model(autoencoder, to_file='autoencoder_graph.png')

# Autoencoder training
X_RNA = X_RNA.astype('float32')
X_Proteomics = X_Proteomics.astype('float32')
Y_RNA = Y_RNA.astype('float32')
Y_Proteomics = Y_Proteomics.astype('float32')
estimator = autoencoder.fit([X_RNA, X_Proteomics],
                          [X_RNA,X_Proteomics],
                            epochs = 100, batch_size = 128,
                            validation_split = 0.2, shuffle = True)
print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')
plt.show()

encoder = Model(inputs= [input_dim_RNA, input_dim_Proteomics],
                outputs = bottleneck)
bottleneck_representation = encoder.predict([X_RNA, X_Proteomics])

# tsne plot
model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123, perplexity = 90, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c = Y_RNA, cmap = 'Spectral')
plt.title('tSNE Autoencoder Data Integration')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.show()

