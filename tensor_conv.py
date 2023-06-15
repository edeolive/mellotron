import tensorflow as tf
import numpy as np
from wave import open
import soundfile
import sys, getopt

class Wave:
    def __init__ (self, data, frame_rate):
        self.data = normalize(data)
        self.frame_rate = frame_rate
    
    def __mul__(self, other):
        x = tf.signal.rfft(self.data, name='fft_output')
        y = tf.signal.rfft(other.data, name='fft_output')
        z_fft = tf.math.multiply(x, y)
        z = tf.signal.irfft(z_fft)
        return Wave(z.numpy().astype(np.float32), self.frame_rate)

    def writeToFile(self, file_path):
        reader = open(file_path, 'w')
        reader.setnchannels(1)
        reader.setsampwidth(2)
        reader.setframerate(self.frame_rate)

        if max(self.data) > 1 or min(self.data) < -1:
            self.data = normalize(self.data)
        reader.writeframes((self.data * 32767).astype(np.int16))
        reader.close()

def normalize(data):
    high, low = abs(max(data)), abs(min(data))
    return data / max(high, low)

def zero_pad_wave(audio, ir):
    if len(audio.data) > len(ir.data):
        zeros = np.zeros(len(audio.data))
        zeros[:len(ir.data)] = ir.data
        ir.data = zeros
    else:
        zeros = np.zeros(len(ir.data))
        zeros[:len(audio.data)] = audio.data
        audio.data = zeros

def convert_wav(file):
    data, samprate = soundfile.read(file)
    soundfile.write(file, data, samprate, subtype='PCM_16')

def file_to_wave(file):
    reader = open(file)
    _, sampwidth, framerate, nframes, _, _ = reader.getparams()
    frames = reader.readframes(nframes)
    reader.close()

    dtypes = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtypes:
        raise ValueError('unsupported sample width')
    
    data = np.frombuffer(frames, dtype=dtypes[sampwidth])
    num_channels = reader.getnchannels()
    if num_channels == 2:
        data = data[::2]

    return Wave(data, framerate)

def conv_reverb(audio_file, ir_file, output_file):
    # convert files to PCM16 wav, then convert to waves
    convert_wav(audio_file)
    convert_wav(ir_file)
    audio = file_to_wave(audio_file)
    ir = file_to_wave(ir_file)
    # convolve using FFT approximation and write output
    zero_pad_wave(audio, ir)
    output_wave = audio * ir
    output_wave.writeToFile(output_file)

def usage():
    print("python tensor_conv.py -a AUDIO_FILE_PATH, -i IMPULSE_FILE_PATH, -o OUTPUT_FILE_PATH\n")
    print("e.g. python tensor_conv.py -a ./samples/1.wav -i ./samples/2.wav -o ./output.wav\n")

def main():
    long_args = ["help", "audio=", "impulse=", "output="]
    audio_file = ''
    ir_file = ''
    output_file = ''
    try:
       opts, args = getopt.getopt(sys.argv[1:], "ha:i:o:", long_args)
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-a", "--audio"):
            audio_file = arg
        elif opt in ("-i", "--impulse"):
            ir_file = arg
        elif opt in ("-o", "--output_file"):
            output_file = arg
        else:
            assert False, "Unhandled option"
    print("Starting convolution...")
    conv_reverb(audio_file, ir_file, output_file)
    print("Convolution written to " + output_file)
    sys.exit()

if __name__ == "__main__":
    main()