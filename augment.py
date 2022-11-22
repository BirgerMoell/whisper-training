from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, SomeOf
import numpy as np
import soundfile as sf

def augment(audio_path, augment_name="-pitch-shift.wav", number_of_augmentations=5):

    samples, sample_rate = sf.read(audio_path)

    augment = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ])

    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    augmented_path = audio_path + augment_name
    sf.write(augmented_path, augmented_samples, sample_rate)
    return augmented_path, len(augmented_samples)

def augment_pitch_shift(audio_path, augment_name="-pitch-shift.wav", number_of_augmentations=5):

    samples, sample_rate = sf.read(audio_path)

    augment = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ])

    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    augmented_path = audio_path + augment_name
    sf.write(augmented_path, augmented_samples, sample_rate)
    return augmented_path, len(augmented_samples)

def augment_time_stretch(audio_path, augment_name="-time-stretch.wav", number_of_augmentations=5):

    samples, sample_rate = sf.read(audio_path)

    augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5, leave_length_unchanged=False),
    ])

    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    augmented_path = audio_path + augment_name
    sf.write(augmented_path, augmented_samples, sample_rate)
    return augmented_path, len(augmented_samples)

def augment_gaussian(audio_path, augment_name="-gaussian.wav", number_of_augmentations=5):

    samples, sample_rate = sf.read(audio_path)

    augment = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.5),
    ])

    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    augmented_path = audio_path + augment_name
    sf.write(augmented_path, augmented_samples, sample_rate)
    return augmented_path, len(augmented_samples)

# def augment_all(audio_path, augment_name="augmented.wav", number_of_augmentations=5):

#     samples, sample_rate = sf.read(audio_path)

#     augmentations = []

#     for i in range(number_of_augmentations):
#         augment = SomeOf(2, None),[
#         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#         TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#         PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#         Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
#         ])

#         augmented_samples = augment(samples=samples, sample_rate=sample_rate)
#         augmented_path = audio_path + augment_name + str(i+1)
#         sf.write(augmented_path, augmented_samples, sample_rate)
#         augmentations.append({"augment_path": augmented_path,
#                            "duration_frames": len(augmented_samples)})
#     return augmentations

def augment_old(audio_path, augment_name="augmented.wav"):

    samples, sample_rate = sf.read(audio_path)

    if "bnt" in audio_path:
        #print("Aphasia person!")
        time_stretch_min_rate= 0.7
        time_stretch_max_rate= 0.9
    elif "vnt" in audio_path:
        #print("Not aphasia found!")
        time_stretch_min_rate= 1.2
        time_stretch_max_rate= 1.8
    else:
        raise ValueError("Invalid path", audio_path)

    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=time_stretch_min_rate, max_rate=time_stretch_max_rate, p=0.5, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    augmented_path = audio_path + augment_name
    sf.write(augmented_path, augmented_samples, sample_rate)
    return augmented_path, len(augmented_samples)

   