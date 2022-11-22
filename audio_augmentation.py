
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

def augment_dataset_noise(batch):

    for i in range(len(batch["audio"])):

        audio = batch["audio"][i]["array"]
        # apply augmentation
        augmented_audio = augment_noise(samples=audio, sample_rate=16000)

        batch["audio"][i]["array"] = augmented_audio

    return batch

def augment_dataset_time_stretch(batch):

    for i in range(len(batch["audio"])):

        audio = batch["audio"][i]["array"]
        # apply augmentation
        augmented_audio = augment_time_stretch(samples=audio, sample_rate=16000)

        batch["audio"][i]["array"] = augmented_audio

    return batch

def augment_dataset_pitch(batch):

    for i in range(len(batch["audio"])):

        audio = batch["audio"][i]["array"]
        # apply augmentation
        augmented_audio = augment_pitch_shift(samples=audio, sample_rate=16000)

        batch["audio"][i]["array"] = augmented_audio

    return batch

augment_noise = Compose([AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1),])
augment_time_stretch = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1, leave_length_unchanged=False),])
augment_pitch_shift = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1),])