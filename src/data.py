import torch
import torch.utils.data
import math
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class VibravoxDataset:
    """Simple paired dataset for Vibravox throat-to-acoustic mapping.

    Returns (throat, acoustic) pairs without noise augmentation.
    """
    def __init__(self,
                 datapair_list,
                 sampling_rate=16_000,
                 segment=None,
                 stride=None,
                 with_id=False,
                 with_text=False):
        self.sampling_rate = sampling_rate
        self.with_id = with_id
        self.with_text = with_text
        assert not self.with_text or self.with_id, "with_id must be True if with_text is True"

        throat_list, acoustic_list = [], []
        for item in datapair_list:
            throat = item["audio.throat_microphone"]['array'].astype('float32')
            acoustic = item["audio.acoustic_microphone"]['array'].astype('float32')
            uid = item["speaker_id"] + "_" + item["sentence_id"]
            text = item["text"]
            length = throat.shape[-1]
            throat_list.append((throat, uid, text, length))
            acoustic_list.append((acoustic, uid, text, length))

        self.throat_set = Audioset(wavs=throat_list, segment=segment, stride=stride)
        self.acoustic_set = Audioset(wavs=acoustic_list, segment=segment, stride=stride)

    def __len__(self):
        return len(self.throat_set)

    def __getitem__(self, index):
        throat_wav, uid, text = self.throat_set[index]
        acoustic_wav, _, _ = self.acoustic_set[index]

        throat = torch.tensor(throat_wav, dtype=torch.float32)
        acoustic = torch.tensor(acoustic_wav, dtype=torch.float32)

        if self.with_text:
            return throat, acoustic, uid, text
        elif self.with_id:
            return throat, acoustic, uid
        else:
            return throat, acoustic


class Audioset:
    def __init__(self, wavs=None, segment=None, stride=None):
        self.wavs = wavs
        self.num_examples = []
        self.segment = segment
        self.stride = stride or segment

        for _, _, _, wav_length in self.wavs:
            if segment is None or wav_length < segment:
                examples = 1
            else:
                examples = int(math.ceil((wav_length - self.segment) / self.stride) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (wav, uid, text, _), examples in zip(self.wavs, self.num_examples):
            if index >= examples:
                index -= examples
                continue

            offset = self.stride * index if self.segment else 0
            num_frames = self.segment if self.segment else len(wav)
            wav = wav[offset:offset + num_frames]
            if self.segment:
                wav = np.pad(wav, (0, num_frames - wav.shape[-1]), 'constant')

            return wav, uid, text

