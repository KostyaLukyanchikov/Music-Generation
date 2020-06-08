import numpy as np
import random
import sys
import random
import time
import os
from music21 import instrument, note, stream, chord, duration, volume


def sample(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds)/temperature
        exp_preds = np.exp(preds)
        preds = exp_preds/np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

def build_sequence(maxlen, note_seq, duration_seq, velocity_seq):
    # Select a text seed at random
    start_index = random.randint(0, len(note_seq) - maxlen - 1)
    
    input_note_seq = note_seq[start_index: start_index + maxlen]
    input_duration_seq = duration_seq[start_index: start_index + maxlen]
    input_velocity_seq = velocity_seq[start_index: start_index + maxlen]

    return input_note_seq, input_duration_seq, input_velocity_seq



def generate_sequence(model, generated_notes_start, generated_durations_start, generated_velocities_start, note_to_int, duration_to_int, velocity_to_int,len_seq, temperature):
    note_sequence = []
    duration_sequence = []
    velocity_sequence = []

    generated_note_sequence = []
    generated_duration_sequence = []
    generated_velocity_sequence = []

    generated_notes = list.copy(generated_notes_start)
    generated_durations = list.copy(generated_durations_start)
    generated_velocities = list.copy(generated_velocities_start)

    notes_input_sequence = []
    durations_input_sequence = []
    velocities_input_sequence = []

    for n, d, v in zip(generated_notes,generated_durations,generated_velocities):
        note_int = note_to_int[n]
        duration_int = duration_to_int[d]
        velocity_int = velocity_to_int[v]

        notes_input_sequence.append(note_int)
        durations_input_sequence.append(duration_int)
        velocities_input_sequence.append(velocity_int)
        #prediction_output.append([n, d])

        note_sequence.append(note_int)
        duration_sequence.append(duration_int)
        velocity_sequence.append(velocity_int)

    for i in range(len_seq):

        prediction_input = [np.array([notes_input_sequence]), np.array([durations_input_sequence]), np.array([velocities_input_sequence])]

        note_pred, dur_pred, vel_pred = model.predict(prediction_input, verbose=0)

        next_note = sample(note_pred[0], temperature)
        next_dur = sample(dur_pred[0], temperature)
        next_vel = sample(vel_pred[0], temperature)

        notes_input_sequence.append(next_note)
        notes_input_sequence = notes_input_sequence[1:]
        durations_input_sequence.append(next_dur)
        durations_input_sequence = durations_input_sequence[1:]
        velocities_input_sequence.append(next_vel)
        velocities_input_sequence = velocities_input_sequence[1:]

        note_sequence.append(next_note)
        duration_sequence.append(next_dur)
        velocity_sequence.append(next_vel)
        
        generated_note_sequence.append(next_note)
        generated_duration_sequence.append(next_dur)
        generated_velocity_sequence.append(next_vel)
    
    whole_pattern = [note_sequence, duration_sequence, velocity_sequence]
    gen_pattern = [generated_note_sequence, generated_duration_sequence, generated_velocity_sequence]

    return whole_pattern, gen_pattern




def create_midi(pattern, int_to_note, int_to_duration, int_to_velocity):

    midi_stream = stream.Stream()

    note_seq, duration_seq, velocity_seq = pattern
    
    for n, d, v in zip(note_seq,duration_seq, velocity_seq):

        note_pattern = int_to_note[n]
        duration_pattern = int_to_duration[d]
        velocity_pattern = int_to_velocity[v]        

        # pattern is a chord
        if ('.' in note_pattern) or note_pattern.isdigit():
            notes_in_chord = note_pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.duration = duration.Duration(duration_pattern)
                new_note.volume.velocity  = velocity_pattern * 4
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)
        elif('rest' in note_pattern):
        # pattern is a rest
            new_note = note.Rest()
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)
        else:
        # pattern is a note
            new_note = note.Note(note_pattern)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.volume.velocity  = velocity_pattern * 4
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)
            
    return midi_stream


def write_midi(midi_out, output_folder='generated'):
    midi_stream = midi_out.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write('midi', fp=os.path.join(output_folder, 'output-' + str(timestr) + '.mid'))