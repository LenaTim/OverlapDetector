import librosa
import numpy as np


class Audio():
    def __init__(self, hp):
        self.hp = hp

    def mfcc(self, y):
        return librosa.feature.mfcc(y=y, sr=self.hp.audio.sr,
                                    n_mfcc=self.hp.audio.mfcc.n_mfcc)

    def stft(self, y):
        return np.abs(librosa.stft(y=y, n_fft=self.hp.audio.stft.n_fft,
                                   hop_length=self.hp.audio.stft.hop_length)).T

    def melspec(self, y):
        return librosa.feature.melspectrogram(y=y, sr=self.hp.audio.sr,
                                              n_mels=self.hp.audio.melspec.n_mfcc,
                                              fmax=self.hp.audio.melspec.fmax)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
