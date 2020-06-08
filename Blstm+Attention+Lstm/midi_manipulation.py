import midi
import numpy as np
from constants import *
import os
import tensorflow as tf
import math
import sys


def midiToNoteStateMatrix(midifile):

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]
    #for i, track in enumerate(pattern):
        #print(pattern[i][1].text)

    statematrix = []
    time = 0

    state = [[0,0,0] for x in range(NUM_NOTES)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0,oldstate[x][2]] for x in range(NUM_NOTES)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < MIN_NOTE) or (evt.pitch >= MAX_NOTE):
                        #print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-MIN_NOTE] = [0, 0, 0]
                        else:
                            state[evt.pitch-MIN_NOTE] = [1, 1, evt.velocity/MAX_VELOCITY]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    #print(evt)
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        #print("Found time signature event {}. Bailing!".format(evt))
                        np.asarray(statematrix)

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1
    #print(time)
    return np.asarray(statematrix)


def noteStateMatrixToMidi(vel_statematrix, name="example"):
    
    statematrix = []
    for i, timestep in enumerate(vel_statematrix):
        timestep_vel = []
        for note in timestep:
            timestep_vel.append([note[0], note[1], int(np.floor(note[2]*MAX_VELOCITY))])
        statematrix.append(timestep_vel)
    
    statematrix = np.asarray(statematrix, dtype='int')
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0,0] for x in range(NUM_NOTES)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(NUM_NOTES):
            n = state[i]
            p = prevstate[i]
            v = n[2]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append([i, v])
            elif n[0] == 1:
                onNotes.append([i, v])
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+MIN_NOTE))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=note[1], pitch=note[0]+MIN_NOTE))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    
    midi.write_midifile("{}.mid".format(name), pattern)
    
def load_midi(file):
    fname = file.split('/')[-1]
    #p = midi.read_midifile(file)
    cache_path = CACHE_DIR + '\\' + fname + '.npy'
    #print(cache_path)
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        #print(p)
        note_seq = midiToNoteStateMatrix(file)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == NUM_NOTES
    assert note_seq.shape[2] == 3
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq