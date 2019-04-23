import pandas
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model
from keras.layers import Embedding, Dropout, Conv1D, Bidirectional, TimeDistributed, LSTM, Dense
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom accuracy checking function
def q3_acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def seq2ngrams(seqs, n=3):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs])

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

def save_results(x, y, y_, plot_path):
    print("---")
    print("Input: " + str(x))
    print("Target: " + str(onehot_to_seq(y, revsere_decoder_index).upper()))
    print("Result: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    
    plt.imshow(y.T, cmap='Blues')
    plt.imshow(y_.T, cmap='Reds', alpha=.5)
    plt.yticks([0, 1, 2, 3], ['Helix', 'Sheet', 'Loop', 'Unknown'])
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()

# The maximum amount of amino acids in a sequence to be analyzed
maxlen_seq = 40

preferred_amino_acid_chunk_size = 11
kernel_size = 11

learning_rate = 0.0005
epochs = 20
drop_out = 0.3

# Read data frame
df = pandas.read_csv('datasets/2018-06-06-ss.cleaned.csv')

# Get sequences that match our parameters
input_seqs, target_seqs = df[['seq', 'sst3']][(df.len <= maxlen_seq) & (~df.has_nonstd_aa)].values.T

# Input grams stores amino acid windows of every sequence in the dataset
input_grams = seq2ngrams(input_seqs, preferred_amino_acid_chunk_size)

# Print the number of sequences to be trained and tested upon
print('Number of sequences analyzed: %d' % (len(input_seqs)))

# Preprocessing
tokenizer_encoder = Tokenizer()
# Input grams is a list containing windows of VARIABLE length for each sequence
# Each window is a 'word' and is encoded as an integer using a dictionary
tokenizer_encoder.fit_on_texts(input_grams)
input_data = tokenizer_encoder.texts_to_sequences(input_grams)
input_data = sequence.pad_sequences(input_data, maxlen=maxlen_seq, padding='post')
# Each list of window-encoded integers is padded with 0s until it becomes 128 integers long
tokenizer_decoder = Tokenizer(char_level=True)
tokenizer_decoder.fit_on_texts(target_seqs)
target_data = tokenizer_decoder.texts_to_sequences(target_seqs)
target_data = sequence.pad_sequences(target_data, maxlen=maxlen_seq, padding='post')
target_data = to_categorical(target_data)

print('Input data shape: ' + str(input_data.shape))
print('Target data shape: ' + str(target_data.shape))

n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

print('Number of different windows: %d' % n_words)
print('Number of possible structures: %d' % n_tags)

# For example:
# The sequence KCK will have three frames: KCK, CK, and K in the case that preferred_amino_acid_chunk_size is 3
# Those frames are then converted into integers and added to a list for the sequence which is added to input_data
# Thus, input_grams could have the following information for example: 3358, 7562, 2, 0...
# The target_data contains the one-hot encoded secondary structure with an additional integer for no structure
# which is used when there is no sequence in the padding.
# The target_data for sequence KCK could thus possibly be: [[0. 1. 0. 0.], [0. 0. 0. 1.], [0. 1. 0. 0.], [1. 0. 0. 0.]...]

# Create the CNN model
cnn_model = Sequential()
# The embedding layer will convert our dictionary integer input into floats
cnn_model.add(Embedding(input_dim=n_words, output_dim=128, input_length=maxlen_seq))
# The 1d convolutional layer has 128 filters that are each 11 elements long
cnn_model.add(Conv1D(128, kernel_size, padding='same', activation='relu'))
# Add a dropout layer to prevent over-fitting (deactivates 30% of input neurons)
cnn_model.add(Dropout(drop_out))
# Add another convolutional layer that now has 64 filters that are still 11 elements long
cnn_model.add(Conv1D(64, kernel_size, padding='same', activation='relu'))
# Add another dropout layer that deactivates 30% of input neurons
cnn_model.add(Dropout(drop_out))
# Final convolution layer that reduces output to 4 classes and has a softmax activation function
cnn_model.add(Conv1D(n_tags, kernel_size, padding='same', activation='softmax'))

# Create the LSTM RNN model
lstm_model = Sequential()
# Add an embedding layer that will convert our integer dictionary sequences into floats
lstm_model.add(Embedding(input_dim=n_words, output_dim=128, input_length=maxlen_seq))
# Add a bidirectional layer which will pass on information from the past and future states to the output
lstm_model.add(Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=drop_out)))
# The TimeDistributed layer will apply the Dense layer to every neuron and thus make the output n_tags
lstm_model.add(TimeDistributed(Dense(n_tags, activation="softmax")))

# Print summaries of both models
cnn_model.summary()
lstm_model.summary()

# Save plots of models
plot_model(cnn_model, to_file='figures/CNN_q3.png')
plot_model(lstm_model, to_file='figures/LSTM_q3.png')

# The Adam optimizer is a way of learning (in this case we create it ourselves to increase the learning rate)
opt = optimizers.Adam(lr=learning_rate)

# Compile both models
cnn_model.compile(optimizer=opt,
          loss='categorical_crossentropy',
          metrics=['accuracy', q3_acc, 'mae'])

lstm_model.compile(optimizer=opt,
          loss='categorical_crossentropy',
          metrics=['accuracy', q3_acc, 'mae'])

revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

N=3

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.4, random_state=0)

# Split sequences 
seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.4, random_state=0)

# Train CNN model
cnn_history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

# Make the CNN model predict both training and testing data and save the plotted results
y_train_pred = cnn_model.predict(X_train[:N])
y_test_pred = cnn_model.predict(X_test[:N])
print('Training CNN')
for i in range(N):
    save_results(seq_train[i], y_train[i], y_train_pred[i], 'figures/CNN_q3_training' + str(i + 1) + '.png')
print('Testing CNN')
for i in range(N):
    save_results(seq_test[i], y_test[i], y_test_pred[i], 'figures/CNN_q3_testing' + str(i + 1) + '.png')

# Plot CNN model training & validation accuracy values
plt.plot(cnn_history.history['acc'])
plt.plot(cnn_history.history['val_acc'])
plt.title('CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/CNN_q3_accuracy.png', bbox_inches='tight')
plt.clf()

# Plot CNN model training & validation loss values
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('CNN Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/CNN_q3_loss.png', bbox_inches='tight')
plt.clf()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.4, random_state=0)

# Split sequences 
seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.4, random_state=0)

# Train LSTM model
lstm_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

# Make the CNN model predict both training and testing data and save the plotted results
y_train_pred = lstm_model.predict(X_train[:N])
y_test_pred = lstm_model.predict(X_test[:N])
print('Training LSTM')
for i in range(N):
    save_results(seq_train[i], y_train[i], y_train_pred[i], 'figures/LSTM_q3_training' + str(i + 1) + '.png')
print('Testing LSTM')
for i in range(N):
    save_results(seq_test[i], y_test[i], y_test_pred[i], 'figures/LSTM_q3_testing' + str(i + 1) + '.png')

# Plot LSTM model training & validation accuracy values
plt.plot(lstm_history.history['acc'])
plt.plot(lstm_history.history['val_acc'])
plt.title('LSTM Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/LSTM_q3_accuracy.png', bbox_inches='tight')
plt.clf()

# Plot LSTM model training & validation loss values
plt.plot(lstm_history.history['loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('LSTM Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figures/LSTM_q3_loss.png', bbox_inches='tight')
plt.clf()
