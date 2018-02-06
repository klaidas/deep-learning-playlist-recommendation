"""""""""
Using Deep Learning to Recommend Music Playlist
Klaidas Urbanavicius

"""""""""

# Other Imports
import random
import os

# Music Class import
import music

# Data Handling Libraries (Librosa for music, Numpy for numpy arrays)
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
from scipy.misc import imread, imsave

# Deep Learning Libraries
import theano
import lasagne
import nolearn

from lasagne import layers
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from lasagne.updates import momentum

# Program EXEC
print "-- Music Playlist Recommendation using Deep Features --\n"

# Get Music Library Path
library_path = raw_input("Music Library Path: ")
library = [musicfile for musicfile in os.listdir(library_path) if (musicfile.endswith(".mp3"))]


# Choosing k training samples 
lib = len(library)
print "Total Samples: ", lib
k = int(round(0.4 * lib))
print "k: ", k
k_chosen = random.sample(range(0, lib), k)
print "Songs Chosen for Training: ", k_chosen
	

# Define Training and Testing Set
training = []
testing = []
for i in range(lib):
	temp = music.Music(library[i])
	if (i in k_chosen) == True:
		training.append(temp)
	else: 
		testing.append(temp)


# Play Songs allowing user to like or dislike
# Dummy Music Player
print "\n0 if dislike"
print "1 if like"

for i in training:
	user_input = raw_input("\n" + str(i.name) + ": ")
	if user_input == "1":
		i.label = 1
	else:
		i.label = 0


# Split data up into sets that will feed into the network
data = []
labels = []

# Getting Input Mel-Spectrograms from music to use in Deep Learning
for i in training:
	## 256 Mels ## 20s = 862 Frames
	y, sr = librosa.core.load(library_path + "/" + i.name)
	segment = y[60*sr : 80*sr]
	spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, n_mels=256)
	log_spectro = librosa.power_to_db(spectrogram ** 2, ref=1.0)
	i.spectrogram = log_spectro
	x=0
	z=21
	for j in range(41):	
		temp = log_spectro[0 : 256, x : z]
		data.append(temp)
		label = i.label
		labels.append(label)
		x = z
		z += 21

	
for i in testing: 
	## 256 Mels ## 20s = 862 Frames
	y, sr = librosa.core.load(library_path + "/" + i.name)
	segment = y[60*sr : 80*sr]
	spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, n_mels=256)
	log_spectro = librosa.power_to_db(spectrogram ** 2, ref=1.0) 
	i.spectrogram = log_spectro



X_train = np.array(data)
y_train = np.array(labels).astype(np.uint8)
X_train = X_train.reshape(-1, 1, 256, 21)


# Deep Learning 
## Model Definition
CNN = NeuralNet(
	layers = [
		("input", layers.InputLayer), # Input Layer
		("conv2d1", layers.Conv2DLayer), # Convolution Layer
		("dropout3", layers.DropoutLayer), # Dropout Layer
		("conv2d2", layers.Conv2DLayer), # Convolution Layer
		("maxpool2", layers.MaxPool2DLayer), # Max Pooling Layer 
		("conv2d3", layers.Conv2DLayer), # Convolution Layer
		("maxpool3", layers.MaxPool2DLayer), # Max Pooling Layer 
		("dropout1", layers.DropoutLayer), # Dropout Layer
		("dense", layers.DenseLayer), # Fully-Connected Layer
		("dropout2", layers.DropoutLayer), # Dropout Layer
		("output", layers.DenseLayer), # Output Layer
	],
	
	# Layers
	## Input Layer
	input_shape = (None, 1, 256, 21),

	## Convolution Layer 1
	conv2d1_num_filters = 32,
	conv2d1_filter_size = (5, 5),
	conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
	conv2d1_W = lasagne.init.GlorotUniform(),

	## Dropout Layer 3
	dropout3_p = 0.5,

	## Convolution Layer 2
	conv2d2_num_filters = 32,
	conv2d2_filter_size = (5, 5),
	conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
	
	## Max Pooling Layer 2
	maxpool2_pool_size = (2, 2),
	
	## Convolution Layer 3
	conv2d3_num_filters = 32,
	conv2d3_filter_size = (5, 5),
	conv2d3_nonlinearity = lasagne.nonlinearities.rectify,

	## Max Pooling Layer 3
	maxpool3_pool_size = (2, 2),

	## Dropout Layer 1
	dropout1_p = 0.5,
	
	## Fully-Connected/ Dense Layer
	dense_num_units = 256,
	dense_nonlinearity = lasagne.nonlinearities.rectify,
	
	## Dropout Layer 2
	dropout2_p = 0.5,
	
	## Output Layer
	output_num_units = 2,
	output_nonlinearity = lasagne.nonlinearities.softmax,

	# Params
	update = momentum, 
	update_learning_rate = 0.0007,
	update_momentum = 0.9,
	max_epochs = 5,
	verbose = 0,
)


## Model Training
CNN.fit(X_train, y_train)

## Feature Extraction
input_var = CNN.layers_["input"].input_var
dense_layer = layers.get_output(CNN.layers_["dense"], deterministic=True)
dense_function = theano.function([input_var], dense_layer)

for i in training:
	features = []
	tmp = []
	x = 0
	z = 21
	for j in range(41):
		temp = i.spectrogram[0 : 256, x : z]
		temp = temp.reshape(1, 1, 256, 21)
		x = z
		z += 21
		arr = dense_function(temp)
		tmp.append(arr)
	count = np.zeros(tmp[0].shape)
	for x in tmp:
		count += x
	count = count / 41
	features = list(count.ravel())
	i.features = features

for i in testing:
	features = []
	tmp = []
	x = 0
	z = 21
	for j in range(41):
		temp = i.spectrogram[0 : 256, x : z]
		temp = temp.reshape(1, 1, 256, 21)
		x = z
		z += 21
		arr = dense_function(temp)
		tmp.append(arr)
	count = np.zeros(tmp[0].shape)
	for x in tmp:
		count += x
	count = count / 41
	features = list(count.ravel())
	i.features = features

# SVM Classifier
## Model Definition and Training
dataSVM = []
labelsSVM = []

for i in training:
	temp = ""
	if i.label == 1:
		temp = "Like"
	elif i.label == 0:
		temp = "Dislike"
	else: 
		temp = "Dislike"
		print "Error"
	dataSVM.append(i.features)
	labelsSVM.append(temp)

print "Labels", labelsSVM 

svm_model = LinearSVC(dual=False, verbose=0)
svm_model.fit(dataSVM, labelsSVM)


## Prediction and Results
liked_set = []
#disliked_set = []
for i in testing:
	plabel = svm_model.predict(i.features)
	if plabel == "Like":
		liked_set.append(i)
#	else:
#		disliked_set.append(i)
		

# Output Recommended Playlist
print "------LIKED SONGS------"
c = 1
for i in liked_set:
	print "\n[", c, "] : ",  i.name
	c += 1

# For Testing Purposes 
#print "------DISLIKED SONGS------"
#b = 1
#for i in disliked_set:
#	print "\n[", b, "] : ",  i.name
#	b += 1

