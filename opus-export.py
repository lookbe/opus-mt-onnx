from transformers import MarianTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os 

# 1. Define the model ID
# Replace with your desired Opus-MT model (e.g., 'Helsinki-NLP/opus-mt-fr-en')
model_id = "Helsinki-NLP/opus-mt-en-id"

# 2. Load the ORTModel and AutoTokenizer
# The `from_pretrained` method handles the conversion to ONNX if a pre-converted ONNX model 
# is not available on the Hugging Face Hub. Use `from_transformers=True` for this.
print(f"Loading and exporting {model_id} to ONNX...")
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export = True)

save_directory = "./my_exported_onnx_model"
model.save_pretrained(save_directory)

print("Model saved.")
