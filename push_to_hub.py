from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./whisper-small-sv",
    repo_type="model",
    repo_id=""
    ignore_patterns="**/logs/*.txt",
)

# kwargs = {
#     "dataset_tags": "mozilla-foundation/common_voice_11_0",
#     "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
#     "language": "sv-SE",
#     "model_name": "Whisper-Small-Swedish-Birger-Moell",  # a 'pretty' name for our model
#     "finetuned_from": "openai/whisper-small",
#     "tasks": "automatic-speech-recognition",
#     "tags": "hf-asr-leaderboard",
# }

# """The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command:"""

# trainer.push_to_hub(**kwargs)