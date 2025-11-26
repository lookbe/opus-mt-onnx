from transformers import AutoTokenizer

# Load the tokenizer you want to recreate
original_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# Save it to a local directory
save_path = "./my_local_tokenizer_files"
original_tokenizer.save_pretrained(save_path)