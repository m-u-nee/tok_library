import numpy as np
import os
from datatrove.pipeline.tokens.merger import load_doc_ends, get_data_reader
from transformers import PreTrainedTokenizerFast

# Configuration
DEFAULT_PRINT_OPTIONS = {
    "print_input_ids": False,
    "print_tokens": False,
    "print_tokens_post": False,
    "print_decoded_text": True,
    "print_lengths": False
}

DEFAULT_RETURN_OPTIONS = {
    "input_ids": True,
    "tokens": True,
    "tokens_post": True,
    "text": True
}

# Utility Functions
def load_tokenized_data_chunks(data_file):
    """
    Loads tokenized data in chunks from a specified file.

    Args:
        data_file (str): Path to the data file without the `.index` extension.

    Returns:
        generator: A generator that applies the decoding function to each chunk of data.
    """
    try:
        with open(f"{data_file}.index", 'rb') as f:
            doc_ends = load_doc_ends(f)
        reader = get_data_reader(open(data_file, 'rb'), doc_ends, nb_bytes=2)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data or index file: {e}")

    def decode_chunk(x):
        return np.frombuffer(x, dtype=np.uint16).astype(int)

    return map(decode_chunk, reader)

def load_tokenizer(tokenizer_path):
    """
    Initializes and returns a PreTrainedTokenizerFast object from the given path.

    Args:
        tokenizer_path (str): Path to the tokenizer file.

    Returns:
        PreTrainedTokenizerFast: Loaded tokenizer.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="[UNK]",
        pad_token="[PAD]",
        clean_up_tokenization_spaces=False  # Disable token space cleanup
    )

def decode_tokens(tokenizer, input_ids, return_options):
    """
    Decodes input_ids using the tokenizer and returns the requested results.

    Args:
        tokenizer (PreTrainedTokenizerFast): Tokenizer object for decoding.
        input_ids (list[int]): List of token IDs.
        return_options (dict): Options dict for specifying what data to return.

    Returns:
        dict: Decoded results including text, tokens, and post-encoded tokens.
    """
    result = {}
    
    # Decode text if requested
    if return_options.get("text", False):
        text = tokenizer.decode(input_ids)
        result["text"] = text
    
    # Convert to tokens if requested
    if return_options.get("tokens", False):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        result["tokens"] = tokens
    
    # Convert post-encoded tokens if requested
    if return_options.get("tokens_post", False):
        input_ids_post = tokenizer.encode(result.get("text", ""), return_tensors="pt")
        tokens_post = tokenizer.convert_ids_to_tokens(input_ids_post[0])
        result["tokens_post"] = tokens_post
    
    return result

def print_decoded_results(result, input_ids, input_ids_post, print_options):
    """
    Handles the printing of various outputs based on the options provided.

    Args:
        result (dict): Processed result containing decoded data.
        input_ids (list[int]): Original input IDs.
        input_ids_post (list[int]): Post-encoded input IDs.
        print_options (dict): Dict specifying what to print.
    """
    if print_options.get("print_input_ids", False):
        print("Original token IDs:", input_ids)
    
    if print_options.get("print_tokens", False):
        print("Tokens:", result.get("tokens", []))
    
    if print_options.get("print_tokens_post", False):
        print("Tokens after re-encoding:", result.get("tokens_post", []))
    
    if print_options.get("print_decoded_text", False):
        print("Decoded text:", result.get("text", ""))
    
    if print_options.get("print_lengths", False):
        print("Length of original token IDs:", len(input_ids))
        print("Length of re-encoded token IDs:", len(input_ids_post))
        print("Number of words in decoded text:", len(result.get("text", "").split()))

def process_tokenized_data(data_file, tokenizer, print_options=None, return_options=None, max_rows=None):
    """
    Processes tokenized data chunks, decodes them, and optionally prints results.

    Args:
        data_file (str): Path to the data file.
        tokenizer (PreTrainedTokenizerFast): Tokenizer object for decoding.
        print_options (dict): Dictionary with options for printing outputs.
        return_options (dict): Dictionary with options for what to return. Default returns everything.
        max_rows (int, optional): Max number of rows to process.

    Returns:
        list or dict: Processed results based on return_options.
    """
    if print_options is None:
        print_options = DEFAULT_PRINT_OPTIONS
    if return_options is None:
        return_options = DEFAULT_RETURN_OPTIONS

    results = []
    data_chunks = load_tokenized_data_chunks(data_file)
    
    for i, input_ids in enumerate(data_chunks):
        # Decode input IDs
        result = decode_tokens(tokenizer, input_ids, return_options)
        input_ids_post = tokenizer.encode(result.get("text", ""), return_tensors="pt")
        
        # Handle printing logic if needed
        if any(print_options.values()):
            print_decoded_results(result, input_ids, input_ids_post[0], print_options)

        results.append(result)
        
        if max_rows is not None and i >= max_rows:
            break

    # Return results based on return_options
    if len(return_options) == 1 and return_options.get("text", False):
        return [res["text"] for res in results]
    
    return results


# Example usage for actual script running
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tokenizer_path = os.path.join(current_dir, '..', 'data', 'tokenizer', 'tokenizer.json')
    data_file = os.path.join(current_dir, '..', 'processed_output', '00000_00000_shuffled.ds')

    # Initialize tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Process and print data
    process_tokenized_data(
        data_file, 
        tokenizer, 
        print_options=DEFAULT_PRINT_OPTIONS
    )