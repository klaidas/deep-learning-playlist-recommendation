# Deep Learning Project for Playlist Recommendation based on a User's current session 

## Dependencies/ Uses:
* Librosa
* Numpy
* SKLearn
* Theano
* Lasagne
* NoLearn

## System:
- Convolutional Neural Network (CNN) - Deep Learning for Feature Extraction.
-- 3 ConvLayers and 2 FC-Layers.
- Support Vector Machine (SVM) - Machine Learning for Classification.
-- Linear, SVC.

## Algorithm/ Data Flow:
* User provides path to music folder/ library.
* System retrieves all music files that end in .mp3 and stores in array.
* System randomly seperates the music files array into training and testing arrays.
* System "plays"/ displays individual songs to the user.
* User labels each song as "liked" or "disliked".
* System reads in each song, picks a 20s segment out 60s into the song and decodes .mp3 files to .wav.
* System uses the .wav of each file to represent into corresponding Mel Spectrograms.
* System scales the spectrograms by amplitude. 
* System proceeds to slice the spectrograms up into smaller pieces of the spectrogram for data augmentation purposes.
* System defines the CNN parameters and proceeds to train the model using the training set. 
* System uses Theano to extract features from each spectrogram slice passed through the model for both sets.
* System appends all the spectrogram slice's features together to represent that music file and reshapes the features accordingly.
* System trains SVM model with extracted features and known labels from user input from the training set.
* System iterates testing set and attempts to predict by classifying each testing song's features as "liked" or "disliked".
