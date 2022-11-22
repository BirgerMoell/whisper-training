from datasets import load_dataset, DatasetDict, concatenate_datasets

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)

#common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train[:1%]+validation[:1%]", use_auth_token=True)
#common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test[:1%]", use_auth_token=True)

print(common_voice)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

print(common_voice)

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="swedish", task="transcribe")

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="swedish", task="transcribe")

print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
# remove all audio files longer than 10 seconds
#common_voice = common_voice.filter(lambda example: len(example["audio"]["array"]) < 5 * 16000, load_from_cache_file=False)

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

## augmentations
#augment = Compose([AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.5),])
#augment = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5, leave_length_unchanged=False),])
#augment = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=0.5),])

# augment data using the audiomentations library
def augment_dataset(batch):

    for i in range(len(batch["audio"])):

        audio = batch["audio"][i]["array"]
        # apply augmentation
        augmented_audio = augment(samples=audio, sample_rate=16000)

        batch["audio"][i]["array"] = augmented_audio

    return batch

# call augment dataset on the training set
augmented = common_voice["train"].map(augment_dataset, batched=True, num_proc=1)
common_voice["train"] = concatenate_datasets([common_voice["train"], augmented])

print(common_voice["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

"""Let's initialise the data collator we've just defined:"""

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-sv",  # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=1,
    max_steps=10,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5,  # set to < max_steps
    eval_steps=5,  # set to < max_steps
    logging_steps=1,  # set to < max_steps
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    #optim="adamw_bnb_8bit",  # set the optimiser!
    optim="adafactor"
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

trainer.train()

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "language": "sv",
    "model_name": "whisper-large-sv",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

# import pdb
# pdb.set_trace()

trainer.push_to_hub(**kwargs)

# from transformers import pipeline
# import gradio as gr

# pipe = pipeline(model="birgermoell/whisper-large")  # change to "your-username/the-name-you-picked"

# def transcribe(audio):
#     text = pipe(audio)["text"]
#     return text

# iface = gr.Interface(
#     fn=transcribe, 
#     inputs=gr.Audio(source="microphone", type="filepath"), 
#     outputs="text",
#     title="Whisper large SV",
#     description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
# )

# iface.launch()
