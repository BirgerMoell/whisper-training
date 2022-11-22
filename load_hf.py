from transformers import pipeline
from transformers import AutoProcessor, pipeline
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("birgermoell/whisper-small-sv-bm")
pipe = pipeline(model="birgermoell/whisper-small-sv-bm")  # change to "your-username/the-name-you-picked"
#pipe = pipeline(model="birgermoell/whisper-small-sv-bm", device=0)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

import time
startTime = time.time()

result = pipe(ds[0]["audio"]["array"])["text"]
print("result:", result)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))