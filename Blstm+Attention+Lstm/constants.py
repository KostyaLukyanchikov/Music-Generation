import os

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 6
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 24
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64
SONG_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

songs = ['mozart_beethoven_albeniz/alb_esp1.mid',
         'mozart_beethoven_albeniz/alb_esp2.mid',
         'mozart_beethoven_albeniz/alb_esp3.mid',
         'mozart_beethoven_albeniz/alb_esp4.mid',
         'mozart_beethoven_albeniz/alb_esp5.mid',
         'mozart_beethoven_albeniz/alb_esp6.mid',
         'mozart_beethoven_albeniz/alb_se1.mid',
         'mozart_beethoven_albeniz/alb_se2.mid',
         'mozart_beethoven_albeniz/alb_se3.mid',
         'mozart_beethoven_albeniz/alb_se4.mid',
         'mozart_beethoven_albeniz/alb_se5.mid',
         'mozart_beethoven_albeniz/alb_se6.mid',
         'mozart_beethoven_albeniz/alb_se7.mid',
         'mozart_beethoven_albeniz/alb_se8.mid',
         'mozart_beethoven_albeniz/appass_1.mid',
         'mozart_beethoven_albeniz/appass_2.mid',
         'mozart_beethoven_albeniz/appass_3.mid',
         'mozart_beethoven_albeniz/beethoven_hammerklavier_1.mid',
         'mozart_beethoven_albeniz/beethoven_hammerklavier_2.mid',
         'mozart_beethoven_albeniz/beethoven_hammerklavier_3.mid',
         'mozart_beethoven_albeniz/beethoven_hammerklavier_4.mid',
         'mozart_beethoven_albeniz/beethoven_les_adieux_1.mid',
         'mozart_beethoven_albeniz/beethoven_les_adieux_2.mid',
         'mozart_beethoven_albeniz/beethoven_les_adieux_3.mid',
         'mozart_beethoven_albeniz/beethoven_opus10_1.mid',
         'mozart_beethoven_albeniz/beethoven_opus10_2.mid',
         'mozart_beethoven_albeniz/beethoven_opus10_3.mid',
         'mozart_beethoven_albeniz/beethoven_opus22_1.mid',
         'mozart_beethoven_albeniz/beethoven_opus22_2.mid',
         'mozart_beethoven_albeniz/beethoven_opus22_3.mid',
         'mozart_beethoven_albeniz/beethoven_opus22_4.mid',
         'mozart_beethoven_albeniz/beethoven_opus90_1.mid',
         'mozart_beethoven_albeniz/beethoven_opus90_2.mid',
         'mozart_beethoven_albeniz/elise.mid',
         'mozart_beethoven_albeniz/mond_1.mid',
         'mozart_beethoven_albeniz/mond_2.mid',
         'mozart_beethoven_albeniz/mond_3.mid',
         'mozart_beethoven_albeniz/mz_311_1.mid',
         'mozart_beethoven_albeniz/mz_311_2.mid',
         'mozart_beethoven_albeniz/mz_311_3.mid',
         'mozart_beethoven_albeniz/mz_330_1.mid',
         'mozart_beethoven_albeniz/mz_330_2.mid',
         'mozart_beethoven_albeniz/mz_330_3.mid',
         'mozart_beethoven_albeniz/mz_331_1.mid',
         'mozart_beethoven_albeniz/mz_331_2.mid',
         'mozart_beethoven_albeniz/mz_331_3.mid',
         'mozart_beethoven_albeniz/mz_332_1.mid',
         'mozart_beethoven_albeniz/mz_332_2.mid',
         'mozart_beethoven_albeniz/mz_332_3.mid',
         'mozart_beethoven_albeniz/mz_333_1.mid',
         'mozart_beethoven_albeniz/mz_333_2.mid',
         'mozart_beethoven_albeniz/mz_333_3.mid',
         'mozart_beethoven_albeniz/mz_545_1.mid',
         'mozart_beethoven_albeniz/mz_545_2.mid',
         'mozart_beethoven_albeniz/mz_545_3.mid',
         'mozart_beethoven_albeniz/mz_570_1.mid',
         'mozart_beethoven_albeniz/mz_570_2.mid',
         'mozart_beethoven_albeniz/mz_570_3.mid',
         'mozart_beethoven_albeniz/pathetique_1.mid',
         'mozart_beethoven_albeniz/pathetique_2.mid',
         'mozart_beethoven_albeniz/pathetique_3.mid',
         'mozart_beethoven_albeniz/waldstein_1.mid',
         'mozart_beethoven_albeniz/waldstein_2.mid',
         'mozart_beethoven_albeniz/waldstein_3.mid']

NUM_SONGS = sum(len(s) for s in songs)