from datasets import load_dataset
from transformers import AutoProcessor, pipeline
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("birgermoell/whisper-small-sv-bm")
model = ORTModelForSpeechSeq2Seq.from_pretrained("birgermoell/whisper-small-sv-bm", from_transformers=True)
speech_recognition_pipeline = pipeline(
 "automatic-speech-recognition",
  model=model,
  feature_extractor=processor.feature_extractor,
  tokenizer=processor.tokenizer,
 )

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

import time
startTime = time.time()
#inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
result = speech_recognition_pipeline(ds[0]["audio"]["array"])
print("result:", result)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))