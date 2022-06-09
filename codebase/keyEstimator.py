"""
handles key estimation for error correction

Author : Adam ALIOUAT https://github.com/adam-at/
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class keyEstimator(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend

        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)

        # chroma_vals est la quantité de chaque classe de hauteur dans l'intervalle de temps
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        # dictionnaire associant les classes de hauteur à leurs intensités dans le morceau
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)}

        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # on utilise ici l'algorithme de Krumhansl-Schmuckler
        # profiles typiques des clés mineurs et majeurs:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # calcule les corrélations entre l'intensité de chaque classe de hauteur
        # dans l'intervalle de temps et les profils ci-dessus.
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # coefficients de corrélation
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

        # nom de toutes les clés (majeur et mineur)
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)},
                         **{keys[i + 12]: self.min_key_corrs[i] for i in range(12)}}

        # la clé déterminé par l'algorithme
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())

        # altkey représente le deuxième meilleur choix pour la clé
        # si la corrélation est proche de celle la clé déterminé
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr

    # affiche la probabilité relative de chaque classe de hauteur
    def print_chroma(self):
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            print(key, '\t', f'{chrom / self.chroma_max:5.3f}')

    # affiche les coefficients de corrélation associés à chaque clé mineur et majeur
    def corr_table(self):
        for key, corr in self.key_dict.items():
            print(key, '\t', f'{corr:6.3f}')

    # affiche la clé déterminé par l'algorithme, la seconde clé la plus proche est donné si elle est assez proche
    def print_key(self):
        print("likely key: ", self.key, ", correlation: ", self.bestcorr, sep='')
        if self.altkey is not None:
            print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')

    # affiche un chromagramme pour le morceau représentant les probabilités de chaque classe de hauteur au cours du temps
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=self.sr, bins_per_octave=24)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(C, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    #retourne un dictionnaire associant les clefs aux coefficients de corrélation correspondant pour le morceau
    def correlations(self):
        self.chroma_max = max(self.chroma_vals)
        cor = [(chrom / self.chroma_max) for key,chrom in self.keyfreqs.items()]
        pitches = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        cor2 = {pitches[i]: cor[i] for i in range(12)}
        return cor2
