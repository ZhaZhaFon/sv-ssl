import numpy as np

from sslsv.data.AudioDataset import AudioDataset


class AudioDatasetMC_4x(AudioDataset):

    def __init__(self, config):
        super().__init__(config)

    def sample_frames(self, audio, frame_length):
        #TODO
        audio_length = audio.shape[1]
        assert audio_length >= 2 * frame_length, \
            "audio_length should >= 2 * frame_length"

        dist = audio_length - 2 * frame_length
        dist = np.random.randint(0, dist + 1)

        lower = frame_length + dist // 2
        upper = audio_length - (frame_length + dist // 2)
        pivot = np.random.randint(lower, upper + 1)

        frame1_from = pivot - dist // 2 - frame_length
        frame1_to = pivot - dist // 2
        frame1 = audio[:, frame1_from:frame1_to]

        frame2_from = pivot + dist // 2
        frame2_to = pivot + dist // 2 + frame_length
        frame2 = audio[:, frame2_from:frame2_to]

        return frame1, frame2