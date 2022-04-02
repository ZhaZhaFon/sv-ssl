import numpy as np
import torch

from sslsv.data.AudioDataset import AudioDataset
from sslsv.data.utils import load_audio

class AudioDatasetMC_2x2(AudioDataset):

    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, i):
        if isinstance(i, int):
            data = load_audio(
                self.files[i],
                frame_length=None,
                min_length=2*self.config.frame_length
            ) # (1, T)
            frame1, frame2 = self.sample_frames(data, self.config.frame_length)
            y = self.labels[i]
        else:
            frame1 = load_audio(self.files[i[0]], self.config.frame_length)
            frame2 = load_audio(self.files[i[1]], self.config.frame_length)
            y = self.labels[i[0]]

        X = np.concatenate((
            self.preprocess_data(frame1),
            self.preprocess_data(frame1),
            self.preprocess_data(frame2),
            self.preprocess_data(frame2)
        ), axis=0)
        X = torch.FloatTensor(X)

        return X, y