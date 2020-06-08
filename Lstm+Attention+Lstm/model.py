import tensorflow

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape, BatchNormalization, Flatten
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model
from seq_self_attention import SeqSelfAttention



def create_network(n_notes, n_durations, n_velocities, embed_size, rnn_units):
    
    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))
    velocities_in = Input(shape = (None,))
    
    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in)
    x3 = Embedding(n_velocities, embed_size)(velocities_in)
    
    x = Concatenate()([x1, x2, x3])
    #x =  Concatenate()([notes_in, durations_in, velocities_in])
    
    x = LSTM(rnn_units, recurrent_dropout=0.3, return_sequences=True)(x)
    x = BatchNormalization()(x)

    x = SeqSelfAttention(attention_activation='sigmoid')(x)
    
    x = LSTM(rnn_units, recurrent_dropout=0.3)(x)
    x = BatchNormalization()(x)
    
    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(x)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(x)
    velocities_out = Dense(n_velocities, activation = 'softmax', name = 'velocity')(x)
    
    model = Model([notes_in, durations_in, velocities_in], [notes_out, durations_out, velocities_out])
    
    optimizer = Adam()
    
    model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  metrics = ['accuracy', 'accuracy', 'accuracy'],
                  optimizer = optimizer)
    
    return model