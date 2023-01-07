import numpy as np
import torch
from pydub import AudioSegment

class Image:
    def __init__(self, path, time_steps=1):
        self.path = path
        self.time_steps = time_steps
        self.image = Image.open(self.path)
        self.image = self.image.resize((224, 224))
        self.image = np.array(self.image)
        self.image = self.image.transpose((2, 0, 1))
        self.image = torch.tensor(self.image, dtype=torch.float32)
        self.image = self.image.unsqueeze(0)
        self.image = self.image.repeat(self.time_steps, 1, 1, 1)

    def __getitem__(self, index):
        return self.image[index]

    def __len__(self):
        return self.time_steps

class Audio:
    def __init__(self, path, time_steps=1):
        self.path = path
        self.time_steps = time_steps
        self.audio = AudioSegment.from_file(self.path)
        self.audio = self.audio.set_frame_rate(16000)
        self.audio = self.audio.set_channels(1)
        self.audio = self.audio.set_sample_width(2)
        self.audio = np.array(self.audio.get_array_of_samples())
        self.audio = torch.tensor(self.audio, dtype=torch.float32)
        self.audio = self.audio.unsqueeze(0)
        self.audio = self.audio.repeat(self.time_steps, 1, 1)

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.time_steps
