import onnxruntime as ort
import numpy as np
import json
import os 
import sentencepiece
from typing import List, Dict, Union, Tuple, Optional

# --- Custom MarianTokenizer Implementation (Shim) ---

class MarianTokenizerShim:
    """Minimal shim to replace the Hugging Face MarianTokenizer."""
    def __init__(self, source_spm: str, target_spm: str, vocab: str, unk_token: str, eos_token: str, pad_token: str, model_max_length: int):
        self.spm_source = sentencepiece.SentencePieceProcessor()
        self.spm_source.Load(source_spm)
        self.spm_target = sentencepiece.SentencePieceProcessor()
        self.spm_target.Load(target_spm)
        
        with open(vocab, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        self.unk_token_id = self.encoder[unk_token]
        self.eos_token_id = self.encoder[eos_token]
        self.pad_token_id = self.encoder[pad_token]
        self.all_special_tokens = [unk_token, eos_token, pad_token]
        self.model_max_length = model_max_length
        self.decoder_map = {v: k for k, v in self.encoder.items()}

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.unk_token_id)
        
    def _tokenize(self, text: str) -> List[str]:
        return self.spm_source.encode(text, out_type=str)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return token_ids + [self.eos_token_id]

    def __call__(self, text: str, truncation: bool = True) -> Dict[str, np.ndarray]:
        """Mimics the Hugging Face MarianTokenizer __call__ method."""
        tokens = self._tokenize(text)
        input_ids = self.build_inputs_with_special_tokens(self.convert_tokens_to_ids(tokens))
        
        if truncation and len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length - 1] + [self.eos_token_id]
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": np.array([input_ids], dtype=np.int64), 
            "attention_mask": np.array([attention_mask], dtype=np.int64)
        }

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Replicates MarianTokenizer's convert_tokens_to_string logic."""
        sp_model = self.spm_target
        current_sub_tokens = []
        out_string = ""
        
        for token in tokens:
            if token in self.all_special_tokens:
                out_string += sp_model.DecodePieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        
        out_string += sp_model.DecodePieces(current_sub_tokens)
        # The Python file used " ", which is likely the unicode underline \u2581 replaced by the SP model.
        # We use the original method's implementation logic.
        return out_string.replace(" ", " ").strip()
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts a sequence of IDs back to a string."""
        tokens = []
        for id in token_ids:
            token = self.decoder_map.get(id, self.unk_token)
            if not skip_special_tokens or token not in self.all_special_tokens:
                tokens.append(token)
            
        return self.convert_tokens_to_string(tokens)

# --- SequenceBiasLogitsProcessor (NumPy Shim) ---

class SequenceBiasLogitsProcessor:
    """Applies an additive bias on sequences to influence next token prediction."""
    def __init__(self, sequence_bias: List[List[Union[List[int], float]]]):
        # Convert List[List[List[int], float]] into Dict[Tuple[int], float] for easy lookup
        self.sequence_bias: Dict[Tuple[int], float] = {tuple(item[0]): float(item[1]) for item in sequence_bias}
        self.length_1_bias: Optional[np.ndarray] = None
        self.prepared_bias_variables = False
        
    def _prepare_bias_variables(self, scores: np.ndarray):
        vocabulary_size = scores.shape[-1]
        
        self.length_1_bias = np.zeros((vocabulary_size,), dtype=np.float32)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                if sequence_ids[-1] >= vocabulary_size:
                    raise ValueError(f"Biased token ID {sequence_ids[-1]} is out of vocab range {vocabulary_size}.")
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True
        
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Applies the sequence bias to the token scores (logits)."""
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        bias = np.zeros_like(scores, dtype=np.float32)
        bias += self.length_1_bias

        # Bias for length > 1
        for sequence_ids_tuple, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids_tuple) == 1:
                continue
            
            sequence_ids = list(sequence_ids_tuple)
            prefix_length = len(sequence_ids) - 1
            
            if prefix_length >= input_ids.shape[1]: 
                continue
                
            last_token = sequence_ids[-1]
            prefix_tokens = np.array(sequence_ids[:-1], dtype=input_ids.dtype)
            
            # Check if the generated sequence ends with the required prefix
            comparison = np.equal(input_ids[:, -prefix_length:], prefix_tokens)
            matching_rows = np.all(comparison, axis=1) 

            # Apply bias to the 'last_token' logit for matching rows
            bias[:, last_token] += np.where(matching_rows, sequence_bias, 0.0)

        return scores + bias

# --- Core Inference Logic ---

def create_initial_decoder_input(tokenizer: MarianTokenizerShim, batch_size: int) -> np.ndarray:
    """Creates initial decoder input_ids (<pad> token)."""
    return np.full((batch_size, 1), tokenizer.pad_token_id, dtype=np.int64)

def run_marian_onnx_inference(
    encoder_session: ort.InferenceSession, 
    decoder_session: ort.InferenceSession, 
    decoder_with_past_session: ort.InferenceSession, 
    tokenizer: MarianTokenizerShim,
    input_text: str, 
    max_length: int,
    logits_processor: Optional[SequenceBiasLogitsProcessor] = None
) -> str:
    """Performs sequence-to-sequence translation using the three ONNX models."""
    
    NUM_LAYERS = 6 
    NUM_PAST_FULL = 4 # Dec K, Dec V, Enc K, Enc V
    NUM_PAST_DECODER_ONLY = 2 # New Dec K, New Dec V

    # 1. Tokenize & Encode
    inputs = tokenizer(input_text, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]

    encoder_outputs = encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    encoder_hidden_states = encoder_outputs[0]

    # 2. Decoder Setup
    current_decoder_input_ids = create_initial_decoder_input(tokenizer, batch_size)
    past_key_values = None
    decoded_token_ids = []
    full_decoded_ids_so_far = current_decoder_input_ids
    
    # 3. Generation Loop
    for _ in range(max_length):
        
        decoder_inputs = {}
        output_names = ["logits"]
        
        if past_key_values is None:
            # First Step: Use full decoder
            session = decoder_session
            decoder_inputs.update({
                "input_ids": current_decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask
            })
            for layer_idx in range(NUM_LAYERS): 
                output_names.extend([
                    f"present.{layer_idx}.decoder.key", f"present.{layer_idx}.decoder.value",      
                    f"present.{layer_idx}.encoder.key", f"present.{layer_idx}.encoder.value"
                ])
            
        else:
            # Subsequent Steps: Use decoder with past
            session = decoder_with_past_session
            decoder_inputs.update({
                "input_ids": current_decoder_input_ids,
                "encoder_attention_mask": attention_mask
            })
            for layer_idx in range(NUM_LAYERS): 
                output_names.extend([f"present.{layer_idx}.decoder.key", f"present.{layer_idx}.decoder.value"])

            for layer_idx, layer_past in enumerate(past_key_values):
                # Add past states as inputs
                decoder_inputs[f"past_key_values.{layer_idx}.decoder.key"] = layer_past[0]
                decoder_inputs[f"past_key_values.{layer_idx}.decoder.value"] = layer_past[1]
                decoder_inputs[f"past_key_values.{layer_idx}.encoder.key"] = layer_past[2]
                decoder_inputs[f"past_key_values.{layer_idx}.encoder.value"] = layer_past[3]

        # Run Decoder Step
        decoder_outputs = session.run(output_names, decoder_inputs)
        logits = decoder_outputs[0]

        # Extract and Update Past States
        new_past_key_values = []
        if past_key_values is None:
            for layer_idx in range(NUM_LAYERS):
                start_index = 1 + layer_idx * NUM_PAST_FULL
                new_past_key_values.append(decoder_outputs[start_index:start_index + NUM_PAST_FULL])
        else:
            for layer_idx in range(NUM_LAYERS):
                start_index = 1 + layer_idx * NUM_PAST_DECODER_ONLY
                new_dec_k = decoder_outputs[start_index]
                new_dec_v = decoder_outputs[start_index + 1]
                
                _, _, static_enc_k, static_enc_v = past_key_values[layer_idx]
                new_past_key_values.append([new_dec_k, new_dec_v, static_enc_k, static_enc_v])
                
        past_key_values = new_past_key_values

        # Select Next Token (Greedy Search)
        next_token_logits = logits[:, -1, :]
        
        processed_logits = logits_processor(full_decoded_ids_so_far, next_token_logits) if logits_processor else next_token_logits

        next_token_id = np.argmax(processed_logits, axis=-1).astype(np.int64)
        
        if next_token_id[0] == tokenizer.eos_token_id:
            break
            
        # Update for Next Step
        decoded_token_ids.append(next_token_id[0].item())
        current_decoder_input_ids = next_token_id.reshape(batch_size, 1)

        full_decoded_ids_so_far = np.concatenate(
            [full_decoded_ids_so_far, current_decoder_input_ids], axis=1
        )

    return tokenizer.decode(decoded_token_ids, skip_special_tokens=True)

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Paths and Configuration ---
    # !!! ADJUST THIS PATH !!!
    LOCAL_DIR = "D:/ai/onnx/opus-mt/id-to-en" 
    MAX_LENGTH = 512
    SOURCE_SENTENCE = "halo apa kabar"
    
    SOURCE_SPM_PATH = os.path.join(LOCAL_DIR, "source.spm")
    TARGET_SPM_PATH = os.path.join(LOCAL_DIR, "target.spm")
    SHARED_VOCAB_PATH = os.path.join(LOCAL_DIR, "vocab.json")

    ENCODER_ONNX_PATH = os.path.join(LOCAL_DIR, "encoder_model.onnx")
    DECODER_ONNX_PATH = os.path.join(LOCAL_DIR, "decoder_model.onnx")
    DECODER_WITH_PAST_ONNX_PATH = os.path.join(LOCAL_DIR, "decoder_with_past_model.onnx")

    try:
        # 1. Load Tokenizer
        tokenizer = MarianTokenizerShim(
            source_spm=SOURCE_SPM_PATH, target_spm=TARGET_SPM_PATH, vocab=SHARED_VOCAB_PATH,
            unk_token="<unk>", eos_token="</s>", pad_token="<pad>", model_max_length=MAX_LENGTH
        )
        
        # 2. Initialize ONNX Sessions
        encoder_session = ort.InferenceSession(ENCODER_ONNX_PATH)
        decoder_session = ort.InferenceSession(DECODER_ONNX_PATH)
        decoder_with_past_session = ort.InferenceSession(DECODER_WITH_PAST_ONNX_PATH)
        
        # 3. Initialize Logits Processor: Prevent generation of <pad> token
        pad_token_id = tokenizer.pad_token_id
        SEQUENCE_BIAS_CONFIG = [
            [[pad_token_id], -np.inf],
        ]
        bias_processor = SequenceBiasLogitsProcessor(SEQUENCE_BIAS_CONFIG)
        
        # 4. Run Inference
        print(f"Source: {SOURCE_SENTENCE}")
        print("-" * 30)
        
        translated_text = run_marian_onnx_inference(
            encoder_session, decoder_session, decoder_with_past_session, 
            tokenizer, SOURCE_SENTENCE, MAX_LENGTH,
            logits_processor=bias_processor
        )
        
        print(f"Translation: **{translated_text}**")

    except Exception as e:
        print(f"\nError during execution: {e}")