from transformers import MarianTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os 

# 1. Define the model ID
# Replace with your desired Opus-MT model (e.g., 'Helsinki-NLP/opus-mt-fr-en')
model_id = "Helsinki-NLP/opus-mt-id-en"

# 2. Load the ORTModel and AutoTokenizer
# The `from_pretrained` method handles the conversion to ONNX if a pre-converted ONNX model 
# is not available on the Hugging Face Hub. Use `from_transformers=True` for this.
print(f"Loading and exporting {model_id} to ONNX...")
save_directory = "my_exported_onnx_model"

# 1. Define the local directory where you saved the ONNX model
local_model_path = "./my_exported_onnx_model"

# 2. Load the model from the local directory (no export=True needed)
# It will load the ONNX files and configuration automatically.
model = ORTModelForSeq2SeqLM.from_pretrained(local_model_path)

# model = ORTModelForSeq2SeqLM.from_pretrained(save_directory, export = True)
# save_directory = "./my_exported_onnx_model"
# model.save_pretrained(save_directory)

local_dir = "./my_local_tokenizer_files/"
source_spm_path = os.path.join(local_dir, "source.spm")
target_spm_path = os.path.join(local_dir, "target.spm")
shared_vocab_path = os.path.join(local_dir, "vocab.json")

tokenizer = MarianTokenizer(source_spm=source_spm_path, target_spm=target_spm_path, vocab=shared_vocab_path)

# 4. Run Inference
input_text = "halo apa kabar"

print(f"\nInput: {input_text}")
test_token = tokenizer(
    input_text, 
    return_tensors="np", 
    padding="longest", 
    truncation=True
)
print("Model loaded as ORTModelForSeq2SeqLM.")

# 3. Create the translation pipeline
# Note the task name format: 'translation_{source_lang}_to_{target_lang}'
task_name = "translation_id_to_en"
onnx_translation_pipeline = pipeline(
    task_name,
    model=model,
    tokenizer=tokenizer
)

# The pipeline handles tokenization, ONNX inference, and decoding automatically
result = onnx_translation_pipeline(
    input_text, 
    num_beams=1,          # Set to 1 to disable beam search
    do_sample=False       # Set to False to disable sampling (greedy approach)
)
# 5. Print the result
translated_text = result[0]['translation_text']
print(f"Output: {translated_text}")