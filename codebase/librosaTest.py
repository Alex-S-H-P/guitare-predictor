import librosa
import numpy

filename = "/home/schoepp/Music/bensound-november.mp3"

y, sr = librosa.load(filename)
n_fft = 1024
hop_length = 512

spec = numpy.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
freqs = librosa.core.fft_frequencies(n_fft=n_fft)
times = librosa.core.frames_to_time(spec[0], sr=sr, n_fft=n_fft, hop_length=hop_length)

print('spectrogram size', spec.shape)

fft_bin = 14
time_idx = 1000

if __name__ == '__main__':
    print('freq (Hz)', freqs[fft_bin], "/", freqs[-1])
    print('time (s)', times[time_idx], "/", times[-1], sum(times)//60, sum(times) % 60)
    print('amplitude', spec[fft_bin, time_idx], spec[0])
    print("amplitude", spec[:, time_idx].shape)
