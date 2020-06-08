import os
import numpy as np
import glob
import sys
import pickle

from tqdm import notebook
from music21 import corpus, converter, instrument, note, chord
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.utils import np_utils


def get_all_files(rootdir):

    potential_files=[]

    for subdir, dirs, files in os.walk(rootdir):
        if (subdir.split('/')[-1] != "Classical"):
            dirname = subdir.split('/')[-1]
            #sys.stdout.write("Parsing dir: %s" % dirname)
            #print()
            #sys.stdout.write("Parsing file number: \n")

        for i, filename in enumerate(files):
            #sys.stdout.write("%s: " % (i+1))
            file = subdir + '\\' + filename
            #print(subdir)
            #print(filename)
            potential_files.append(file)

        #print()
    return potential_files


def all_scores(music_list):
    
    print(len(music_list), 'files in total')
    
    notes = []
    durations = []
    velocities =  []
    
    for i, file in enumerate(notebook.tqdm(music_list)):
        print(i+1, "Parsing %s" % file)
        original_score = converter.parse(file)
        
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(original_score)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            print("found flat")
            notes_to_parse = original_score.flat.notes

        for element in notes_to_parse:              
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                durations.append(element.duration.quarterLength)
                velocities.append(int(np.floor(element.volume.velocity/4)))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                durations.append(element.duration.quarterLength)
                velocities.append(int(np.floor(element.volume.velocity/4)))
            elif isinstance(element, note.Rest):
                notes.append(str(element.name))
                durations.append(element.duration.quarterLength)
                velocities.append(0)
                
    return notes, durations, velocities


def get_distinct(elements):
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)


def create_lookups(element_names):
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))
    return (element_to_int, int_to_element)


def create_distincts_and_lookups(notes, durations, velocities):
    note_names, n_notes = get_distinct(notes)
    duration_names, n_durations = get_distinct(durations)
    velocity_names, n_velocities = get_distinct(velocities)
    
    distincts = [note_names, n_notes, duration_names, n_durations, velocity_names, n_velocities]
    
    note_to_int, int_to_note = create_lookups(note_names)
    duration_to_int, int_to_duration = create_lookups(duration_names)
    velocity_to_int, int_to_velocity = create_lookups(velocity_names)
    
    lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration, velocity_to_int, int_to_velocity]
    
    return distincts, lookups



def prepare_sequences(notes, durations, velocities, lookups, distincts, seq_len):
    
    note_to_int, int_to_note, duration_to_int, int_to_duration, velocity_to_int, int_to_velocity = lookups
    note_names, n_notes, durations_names, n_durations, velocity_names, n_velocities = distincts
    
    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []
    velocities_network_input = []
    velocities_network_output = []
    
    for i in range(len(notes) - seq_len):

        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])
        
        velocities_sequence_in = velocities[i:i + seq_len]
        velocities_sequence_out = velocities[i + seq_len]
        velocities_network_input.append([velocity_to_int[char] for char in velocities_sequence_in])
        velocities_network_output.append(velocity_to_int[velocities_sequence_out])
        
    n_patterns = len(notes_network_input)
    
    #reshape into LSTM format
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    velocities_network_input = np.reshape(velocities_network_input, (n_patterns, seq_len))
    
    notes_network_input = notes_network_input
    durations_network_input = durations_network_input
    velocities_network_input = velocities_network_input
    
    network_input = [notes_network_input, durations_network_input, velocities_network_input]
    
    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes = n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes = n_durations)
    velocities_network_output = np_utils.to_categorical(velocities_network_output, num_classes = n_velocities)
    
    network_output = [notes_network_output, durations_network_output, velocities_network_output]
    
    return (network_input, network_output)

def make_callbacks_list(weights_folder, logs_base_dir, history_folder):
    checkpoint1 = ModelCheckpoint(
        os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    checkpoint2 = ModelCheckpoint(
        os.path.join(weights_folder, "weights.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='loss'
        , restore_best_weights=True
        , patience = 10
    )

    tensorboard = TensorBoard(
        log_dir= logs_base_dir,
        profile_batch=0
    )
    
    csv_logger = CSVLogger(
        os.path.join(history_folder, "model_history_log.csv"), 
        append=True
    )

    callbacks_list = [
        checkpoint1
        , checkpoint2
        , early_stopping
        , tensorboard
        , csv_logger
     ]
    
    return callbacks_list

def makeWindowsCmdPath(path):
    return '\"' + str(path) + '\"'