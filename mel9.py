import fluidsynth
import numpy as np
import tensorflow as tf

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from enum import Enum
from wave import open

SOUNDS = Enum('Sounds', ['Cello', 'Flute', 'Strings'])

def normalize(data):
    high, low = abs(max(data)), abs(min(data))
    return data / max(high, low)

class Mellotron:
    def __init__(self, instrument=SOUNDS.Cello):
        self.model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
        self.instrument = instrument

    def _midify(self, audio):
        _, midi_data, _ = predict(audio, self.model)
        return midi_data

    def transform(self, audio):
        midi = self._midify(audio)
        # Change line below for different soundfonts
        audio_data = midi.fluidsynth('C:\ProgramData\soundfonts\Mellotron.sf2')
        self._writeToFile('./output.wav', audio_data)

    def _writeToFile(self, file_path, data):
        reader = open(file_path, 'w')
        reader.setnchannels(1)
        reader.setsampwidth(2)
        reader.setframerate(self.frame_rate)

        if max(self.data) > 1 or min(self.data) < -1:
            data = normalize(data)
        reader.writeframes((data * 32767).astype(np.int16))
        reader.close()

mel = Mellotron()
mel.transform('./samples/1.wav')