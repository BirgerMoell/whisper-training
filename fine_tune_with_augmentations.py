from datasets import load_dataset, DatasetDict, concatenate_datasets

common_voice = DatasetDict()

#common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
#common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train[:1%]+validation[:1%]", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test[:1%]", use_auth_token=True)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

"""Re-loading the first audio sample in the Common Voice dataset will resample 
it to the desired sampling rate:
"""

print(common_voice["train"][0])

## add augmentations
from audio_augmentation import augment_dataset_noise, augment_dataset_time_stretch, augment_dataset_pitch

# call augment dataset on the training set
augmented_noise = common_voice["train"].map(augment_dataset_noise, batched=True, num_proc=1)
augmented_pitch = common_voice["train"].map(augment_dataset_pitch, batched=True, num_proc=1)
augmented_time_stretch = common_voice["train"].map(augment_dataset_time_stretch, batched=True, num_proc=1)
common_voice["train"] = concatenate_datasets([common_voice["train"], augmented_noise, augmented_pitch, augmented_time_stretch])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

"""We can apply the data preparation function to all of our training examples using dataset's `.map` method:"""

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

"""### Load a Pre-Trained Checkpoint

Now let's load the pre-trained Whisper `small` checkpoint. Again, this 
is trivial through use of 🤗 Transformers!
"""

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

"""Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):"""

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

"""### Define the Training Configuration

In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
"""

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sv-bm",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

"""**Note**: if one does not want to upload the model checkpoints to the Hub, 
set `push_to_hub=False`.

We can forward the training arguments to the 🤗 Trainer along with our model,
dataset, data collator and `compute_metrics` function:
"""

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

"""Our best WER is 32.0% - not bad for 8h of training data! We can submit our checkpoint to the [`hf-speech-bench`](https://huggingface.co/spaces/huggingface/hf-speech-bench) on push by setting the appropriate key-word arguments (kwargs):"""

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "language": "sv",
    "model_name": "WhispersmallSwedishBirgerMoell",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

"""The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command:"""

trainer.push_to_hub(**kwargs)