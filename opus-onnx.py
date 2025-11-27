import onnxruntime as ort
import numpy as np
# from transformers import MarianTokenizer # REMOVED DEPENDENCY
import json
import os 
import sentencepiece # NEW DEPENDENCY for core tokenization logic
from typing import List, Dict, Union, Any, Tuple, Optional

# --- Define Paths ---
local_dir = "./my_local_tokenizer_files/"
source_spm_path = os.path.join(local_dir, "source.spm")
target_spm_path = os.path.join(local_dir, "target.spm")
shared_vocab_path = os.path.join(local_dir, "vocab.json")

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-id-en" 
ENCODER_ONNX_PATH = "my_exported_onnx_model/encoder_model.onnx"
DECODER_ONNX_PATH = "my_exported_onnx_model/decoder_model.onnx"
DECODER_WITH_PAST_ONNX_PATH = "my_exported_onnx_model/decoder_with_past_model.onnx"
MAX_LENGTH = 512
SOURCE_SENTENCE = "halo apa kabar"

# Constant used for cleaning up SentecePiece output
SPIECE_UNDERLINE = " "

# --- Custom MarianTokenizer Implementation (Shim) ---

class MarianTokenizerShim:
    """
    Minimal shim to replace the Hugging Face MarianTokenizer dependency.
    It replicates the essential encoding and decoding logic using sentencepiece.
    """
    def __init__(
        self,
        source_spm: str,
        target_spm: str,
        vocab: str, # path to vocab.json
        unk_token: str,
        eos_token: str,
        pad_token: str,
        model_max_length: int
    ):
        # 1. Load SentencePiece models
        self.spm_source = sentencepiece.SentencePieceProcessor()
        self.spm_source.Load(source_spm)
        self.spm_target = sentencepiece.SentencePieceProcessor()
        self.spm_target.Load(target_spm)
        
        # We only need the target SPM for decoding, and Marian uses the target SPM for decoding.
        self.current_spm = self.spm_target 

        # 2. Load Vocabulary (token-to-ID mapping)
        with open(vocab, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        
        # 3. Special Tokens setup
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        # 4. Get IDs for quick access
        if unk_token not in self.encoder: raise KeyError(f"'{unk_token}' not in vocab")
        if eos_token not in self.encoder: raise KeyError(f"'{eos_token}' not in vocab")
        if pad_token not in self.encoder: raise KeyError(f"'{pad_token}' not in vocab")
        
        self.unk_token_id = self.encoder[unk_token]
        self.eos_token_id = self.encoder[eos_token]
        self.pad_token_id = self.encoder[pad_token]
        
        # 5. List of all special tokens (for decoding logic)
        self.all_special_tokens = [unk_token, eos_token, pad_token]
        self.model_max_length = model_max_length
        
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an ID (int) using the shared vocabulary."""
        return self.encoder.get(token, self.unk_token_id)
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text using the source SentencePiece model.
        In Opus-MT, tokenization happens on the raw text without language code handling for the source.
        """
        # Encode as pieces (tokens)
        pieces = self.spm_source.encode(text, out_type=str)
        return pieces

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a sequence of tokens into a sequence of IDs."""
        return [self._convert_token_to_id(token) for token in tokens]

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id (as per MarianTokenizer source)."""
        return token_ids + [self.eos_token_id]

    def __call__(
        self, 
        text: str, 
        return_tensors: str = "np", 
        padding: str = "longest", 
        truncation: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Mimics the Hugging Face MarianTokenizer __call__ method for simple, single-sentence input.
        """
        # 1. Tokenize and Convert to IDs
        tokens = self._tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        
        # 2. Add special tokens (append EOS)
        input_ids = self.build_inputs_with_special_tokens(token_ids)
        
        # 3. Truncation (simple implementation)
        if truncation and len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length - 1] + [self.eos_token_id]
        
        # 4. Padding/Attention Mask (simple implementation for one sentence)
        max_len = len(input_ids)
        
        # Pad up to max_len (trivial for single sentence)
        padded_ids = input_ids
        
        # Attention Mask
        attention_mask = [1] * len(padded_ids)
        
        # 5. Finalize as numpy arrays (batch_size=1)
        input_ids_np = np.array([padded_ids], dtype=np.int64)
        attention_mask_np = np.array([attention_mask], dtype=np.int64)

        return {
            "input_ids": input_ids_np, 
            "attention_mask": attention_mask_np
        }

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Replicates MarianTokenizer's convert_tokens_to_string logic using the target SPM (spm_target).
        This handles special tokens and SentencePiece's underline character cleanup.
        """
        sp_model = self.spm_target
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                # Decode accumulated pieces, append special token, and add a space
                out_string += sp_model.DecodePieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        
        # Decode any remaining sub-tokens
        out_string += sp_model.DecodePieces(current_sub_tokens)
        
        # Clean up the SPIECE_UNDERLINE
        out_string = out_string.replace(SPIECE_UNDERLINE, " ")
        return out_string.strip()
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a sequence of IDs back to a string.
        """
        # Reverse map ID to token string
        # We need the decoder map from tokenization_marian, which is {v: k for k, v in self.encoder.items()}
        decoder = {v: k for k, v in self.encoder.items()}
        
        tokens = []
        for id in token_ids:
            token = decoder.get(id, self.unk_token)
            if skip_special_tokens and token in self.all_special_tokens:
                continue
            tokens.append(token)
            
        return self.convert_tokens_to_string(tokens)

# ----------------------------------------------------
# --- SequenceBiasLogitsProcessor (NumPy Shim) ---
# ----------------------------------------------------

class SequenceBiasLogitsProcessor:
    """
    Reimplementation of SequenceBiasLogitsProcessor using NumPy for ONNX Runtime.
    Applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it.
    """
    def __init__(self, sequence_bias: List[List[Union[List[int], float]]]):
        # Convert List[List[...]] format to Dict[Tuple[int], float] for easier lookup
        self.sequence_bias: Dict[Tuple[int], float] = self._convert_list_arguments_into_dict(sequence_bias)
        self.length_1_bias: Optional[np.ndarray] = None
        self.prepared_bias_variables = False
        
    def _convert_list_arguments_into_dict(self, sequence_bias: List[List[Union[List[int], float]]]) -> Dict[Tuple[int], float]:
        """Converts the sequence_bias input format to a dictionary of token tuples to biases."""
        converted_bias = {}
        for item in sequence_bias:
            # item is [token_list, bias_value]
            token_list = item[0]
            bias_value = item[1]
            # Key must be a hashable type (tuple)
            converted_bias[tuple(token_list)] = float(bias_value)
        return converted_bias

    def _prepare_bias_variables(self, scores: np.ndarray):
        """Precomputes the bias for length=1 sequences and validates tokens."""
        # scores shape: (batch_size, vocab_size) - we only need vocab_size
        vocabulary_size = scores.shape[-1]

        # 1. Check biased tokens out of bounds (simplified check)
        invalid_biases = []
        for sequence_ids in self.sequence_bias.keys():
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # 2. Precompute the bias for length 1 sequences.
        self.length_1_bias = np.zeros((vocabulary_size,), dtype=np.float32)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                # The bias is applied to the single token ID
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True
        
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Applies the sequence bias to the token scores (logits)."""
        # input_ids shape: (batch_size, current_sequence_length)
        # scores shape: (batch_size, vocab_size)

        # 1 - Prepares the bias tensors.
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - prepares an empty bias to add (using float32 for consistency with ORT outputs)
        bias = np.zeros_like(scores, dtype=np.float32)

        # 3 - include the bias from length = 1
        bias += self.length_1_bias

        # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
        
        for sequence_ids_tuple, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids_tuple) == 1:  # the sequence is of length 1, already applied
                continue
            
            sequence_ids = list(sequence_ids_tuple)
            prefix_length = len(sequence_ids) - 1
            
            # Skip if the sequence prefix is longer than the current context
            if prefix_length >= input_ids.shape[1]: 
                continue
                
            last_token = sequence_ids[-1]
            
            # NumPy equivalent of torch.eq(..., ...).prod(dim=1)
            # Check if the generated sequence ends with the required prefix
            comparison = np.equal(
                input_ids[:, -prefix_length:],
                np.array(sequence_ids[:-1], dtype=input_ids.dtype)
            )
            
            # np.all is the equivalent of .prod(dim=1) for boolean arrays
            matching_rows = np.all(comparison, axis=1) 

            # Apply bias:
            # We use np.where to select the bias or 0.0 based on matching_rows
            bias[:, last_token] += np.where(
                matching_rows,
                np.array(sequence_bias, dtype=np.float32),
                np.array(0.0, dtype=np.float32),
            )

        # 5 - apply the bias to the scores
        scores_processed = scores + bias
        return scores_processed


# --- Helper Functions ---

def create_initial_decoder_input(tokenizer: MarianTokenizerShim, batch_size: int):
    """
    Creates the initial decoder input_ids and attention_mask.
    Marian models (like Opus-MT) typically use the <pad> token as the starting token.
    """
    start_token_id = tokenizer.pad_token_id
    
    # Initial input_ids shape: (batch_size, 1)
    decoder_input_ids = np.full(
        (batch_size, 1), 
        start_token_id, 
        dtype=np.int64
    )
    # Initial attention_mask shape: (batch_size, 1)
    decoder_attention_mask = np.ones(
        (batch_size, 1), 
        dtype=np.int64
    )
    return decoder_input_ids, decoder_attention_mask

def run_marian_onnx_inference(
    encoder_session: ort.InferenceSession, 
    decoder_session: ort.InferenceSession, 
    decoder_with_past_session: ort.InferenceSession, 
    tokenizer: MarianTokenizerShim, # Changed type hint
    input_text: str, 
    max_length: int,
    logits_processor: Optional[SequenceBiasLogitsProcessor] = None # NEW ARGUMENT
):
    """Performs sequence-to-sequence translation using the three ONNX models."""
    
    # NOTE: These values are specific to the opus-mt-en-de architecture.
    NUM_LAYERS = 6 
    NUM_PAST_TENSORS_PER_LAYER_FULL = 4 # Dec K, Dec V, Enc K, Enc V
    NUM_PAST_TENSORS_PER_LAYER_DECODER_ONLY = 2 # New Dec K, New Dec V
    
    print(f"✅ Detected {NUM_LAYERS} decoder layers.")
    
    # 2. Tokenize Input
    # The tokenizer.__call__ returns a dict of numpy arrays
    inputs = tokenizer(
        input_text, 
        return_tensors="np", 
        padding="longest", 
        truncation=True
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    batch_size = input_ids.shape[0]

    # 3. Run Encoder
    encoder_inputs = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    encoder_hidden_states = encoder_outputs[0]

    # --- Start Decoding Loop ---
    
    # 4. Prepare Initial Decoder Inputs
    decoder_input_ids, _ = create_initial_decoder_input(
        tokenizer, 
        batch_size
    )

    current_decoder_input_ids = decoder_input_ids
    past_key_values = None # Stores list of [dec_k, dec_v, enc_k, enc_v] for each layer
    decoded_token_ids = []
    
    # NEW: Accumulate the full sequence generated so far (starts with <pad>)
    full_decoded_ids_so_far = decoder_input_ids 
    
    for i in range(max_length):
        
        # 5. Define Inputs/Session based on the step
        decoder_inputs = {}
        output_names = ["logits"]
        
        if past_key_values is None:
            # === First Step (i=0): Use DECODER_ONNX_PATH ===
            session = decoder_session
            
            # Inputs: full context for the initial forward pass
            decoder_inputs["input_ids"] = current_decoder_input_ids
            decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
            decoder_inputs["encoder_attention_mask"] = attention_mask
            
            # Outputs: Logits + ALL 4 tensors per layer (Dec K/V, Enc K/V)
            for layer_idx in range(NUM_LAYERS): 
                output_names.extend([
                    f"present.{layer_idx}.decoder.key",        
                    f"present.{layer_idx}.decoder.value",      
                    f"present.{layer_idx}.encoder.key",  
                    f"present.{layer_idx}.encoder.value",
                ])
            
        else:
            # === Subsequent Steps (i>0): Use DECODER_WITH_PAST_ONNX_PATH ===
            session = decoder_with_past_session
            
            # Inputs: only the last predicted token
            decoder_inputs["input_ids"] = current_decoder_input_ids # (1, 1)
            decoder_inputs["encoder_attention_mask"] = attention_mask # Still needed by the model
            
            # Outputs: Logits + only the 2 updated decoder tensors per layer
            for layer_idx in range(NUM_LAYERS): 
                output_names.extend([
                    f"present.{layer_idx}.decoder.key",    
                    f"present.{layer_idx}.decoder.value",  
                ])

            # CRITICAL FIX: Add past key-value states to the inputs (all 4 tensors per layer)
            for layer_idx, layer_past in enumerate(past_key_values):
                # layer_past is [dec_k, dec_v, enc_k, enc_v] from previous step
                past_d_k, past_d_v, past_e_k, past_e_v = layer_past
                
                # These are the *INPUT* names expected by DECODER_WITH_PAST
                decoder_inputs[f"past_key_values.{layer_idx}.decoder.key"] = past_d_k
                decoder_inputs[f"past_key_values.{layer_idx}.decoder.value"] = past_d_v
                # FIX: Include the static ENCODER K/V states as inputs
                decoder_inputs[f"past_key_values.{layer_idx}.encoder.key"] = past_e_k
                decoder_inputs[f"past_key_values.{layer_idx}.encoder.value"] = past_e_v

        # 6. Run Decoder Step
        decoder_outputs = session.run(output_names, decoder_inputs)
        logits = decoder_outputs[0]

        # 7. Extract and Update Past States
        new_past_key_values = []
        
        if past_key_values is None:
            # First Step: Outputs are logits + (4 tensors * NUM_LAYERS)
            for layer_idx in range(NUM_LAYERS):
                start_index = 1 + layer_idx * NUM_PAST_TENSORS_PER_LAYER_FULL
                # Extract all 4 tensors (dec_k, dec_v, enc_k, enc_v)
                layer_past = decoder_outputs[start_index:start_index + NUM_PAST_TENSORS_PER_LAYER_FULL]
                new_past_key_values.append(layer_past)
        else:
            # Subsequent Steps: Outputs are logits + (2 tensors * NUM_LAYERS)
            for layer_idx in range(NUM_LAYERS):
                start_index = 1 + layer_idx * NUM_PAST_TENSORS_PER_LAYER_DECODER_ONLY
                # Extract 2 NEW decoder tensors
                new_dec_k = decoder_outputs[start_index]
                new_dec_v = decoder_outputs[start_index + 1]
                
                # Reuse the static encoder K/V from the previous step (index 2 and 3)
                _, _, static_enc_k, static_enc_v = past_key_values[layer_idx]
                
                # New list contains: [new_dec_k, new_dec_v, static_enc_k, static_enc_v]
                new_past_key_values.append([new_dec_k, new_dec_v, static_enc_k, static_enc_v])
                
        past_key_values = new_past_key_values

        # 8. Select Next Token (Greedy Search)
        # Logits shape: (batch_size, sequence_length_target, vocab_size)
        next_token_logits = logits[:, -1, :] # Logits for the last (newest) token
        
        # --- NEW LOGIC: Apply Logits Processor ---
        processed_logits = next_token_logits
        if logits_processor is not None:
            # The processor operates on the accumulated sequence so far and the last step's logits
            processed_logits = logits_processor(full_decoded_ids_so_far, next_token_logits)

        # Apply argmax (Greedy Search) to the processed logits
        next_token_id = np.argmax(processed_logits, axis=-1).astype(np.int64)
        
        # Check for End-of-Sentence token
        if next_token_id[0] == tokenizer.eos_token_id:
            break
            
        # 9. Update for Next Step
        decoded_token_ids.append(next_token_id[0].item())
        
        # The next decoder input is the token we just predicted
        current_decoder_input_ids = next_token_id.reshape(batch_size, 1)

        # NEW: Update the accumulated sequence for the next processor step
        full_decoded_ids_so_far = np.concatenate(
            [full_decoded_ids_so_far, current_decoder_input_ids], axis=1
        )


    # 10. Decode Output Tokens
    final_output = tokenizer.decode(
        decoded_token_ids, 
        skip_special_tokens=True
    )
    return final_output

# --- Main Execution Block ---

if __name__ == "__main__":
    print("🚀 Initializing Marian ONNX Inference (using Shim)...")
    try:
        # Check for existence of tokenizer files before attempting to load
        if not all(os.path.exists(p) for p in [source_spm_path, target_spm_path, shared_vocab_path]):
             print("\n❌ Error: Local tokenizer files not found.")
             print(f"Please ensure '{local_dir}' contains 'source.spm', 'target.spm', and 'vocab.json'.")
             raise FileNotFoundError("Missing tokenizer files")

        # 1. Load Tokenizer - USING CUSTOM SHIM
        tokenizer = MarianTokenizerShim(
            source_spm=source_spm_path, 
            target_spm=target_spm_path,
            vocab=shared_vocab_path,
            unk_token="<unk>",
            eos_token="</s>",
            pad_token="<pad>",
            model_max_length=MAX_LENGTH
        )
        
        # 2. Initialize ONNX Sessions
        encoder_session = ort.InferenceSession(ENCODER_ONNX_PATH)
        decoder_session = ort.InferenceSession(DECODER_ONNX_PATH)
        decoder_with_past_session = ort.InferenceSession(DECODER_WITH_PAST_ONNX_PATH)
        
        # 3. Initialize Logits Processor (Pad Token Bias)
        
        # Get the ID of the pad token dynamically
        pad_token_id = tokenizer.pad_token_id
        
        # Define the sequence bias configuration to apply a very large negative bias 
        # to the pad token to prevent its generation (mimicking the image's intent).
        # np.NINF is used for negative infinity.
        SEQUENCE_BIAS_CONFIG = [
            [[pad_token_id], -np.inf],
        ]
        
        print(f"\n💡 Applying sequence bias: Preventing generation of PAD token (ID: {pad_token_id})")
        bias_processor = SequenceBiasLogitsProcessor(SEQUENCE_BIAS_CONFIG)
        
        # 4. Run Inference
        print(f"\nSource: {SOURCE_SENTENCE}")
        print("-" * 30)
        
        translated_text = run_marian_onnx_inference(
            encoder_session, 
            decoder_session, 
            decoder_with_past_session, 
            tokenizer, 
            SOURCE_SENTENCE, 
            MAX_LENGTH,
            logits_processor=bias_processor # PASS THE PROCESSOR HERE
        )
        
        print("-" * 30)
        print(f"Translation: **{translated_text}**")

    except FileNotFoundError:
        print(f"\n❌ Error: One or more ONNX model files or tokenizer files not found. Check file paths.")
        print(f" - {ENCODER_ONNX_PATH}")
        print(f" - {DECODER_ONNX_PATH}")
        print(f" - {DECODER_WITH_PAST_ONNX_PATH}")
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
    except Exception as e:
        # Check for ORT error specific to missing inputs
        if "Missing required input" in str(e) or "Input must be specified" in str(e):
             print("\n❌ ONNX Runtime Input Error: Double-check the exact input names for 'decoder_with_past_model.onnx'.")
             print(f"   Specific error: {e}")
        else:
             print(f"\n❌ An unexpected error occurred during inference: {e}")