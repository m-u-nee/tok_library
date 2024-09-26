import numpy as np
from datatrove.pipeline.tokens.merger import load_doc_ends, get_data_reader
from transformers import PreTrainedTokenizerFast
import os

def read_tokenized_data(data_file):
    """
    Reads tokenized data from a specified file.

    Args:
        data_file (str): Path to the data file without the `.index` extension.

    Returns:
        map: A map object that applies the decoding function to each chunk of data read.
    """
    with open(f"{data_file}.index", 'rb') as f:
        doc_ends = load_doc_ends(f)

    reader = get_data_reader(open(data_file, 'rb'), doc_ends, nb_bytes=2)

    def decode(x): 
        return np.frombuffer(x, dtype=np.uint16).astype(int)

    return map(decode, reader)

def initialize_tokenizer(tokenizer_path):
    """
    Initializes and returns a tokenizer from the given path.
    
    Args:
        tokenizer_path (str): Path to the tokenizer file.

    Returns:
        PreTrainedTokenizerFast: Loaded tokenizer.
    """
    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_path),
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="[UNK]",
        pad_token="[PAD]", 
        clean_up_tokenization_spaces=False  # Disable token space cleanup

    )

def process_data(data_file, tokenizer, print_options, return_options=None, max_rows=None):
    """
    Processes the tokenized data, decodes it, and optionally prints results.
    
    Args:
        data_file (str): Path to the data file.
        tokenizer (PreTrainedTokenizerFast): Tokenizer object for decoding.
        print_options (dict): Dictionary with options for printing outputs.
        return_options (dict): Dictionary with options for what to return. Default returns everything.
        
    Returns:
        list or dict: Processed results based on return_options.
    """
    # Default return options
    if return_options is None:
        return_options = {
            "input_ids": True,
            "tokens": True,
            "tokens_post": True,
            "text": True
        }
        
    # Check if all print options are False, to avoid unnecessary processing
    if not any(print_options.values()):
        print_flag = False
    else:
        print_flag = True
        
    results = []
    for i, input_ids in enumerate(read_tokenized_data(data_file)):
         
        result = {}

        # Decode only if required by return_options
        if return_options.get("text", False):
            text = tokenizer.decode(input_ids)
            result["text"] = text
        
        # Process tokens if requested in return_options
        if return_options.get("tokens", False):
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            result["tokens"] = tokens
        
        # Process post-encoded input IDs and tokens if requested
        if return_options.get("tokens_post", False) or print_flag:
            input_ids_post = tokenizer.encode(result.get("text", ""), return_tensors="pt")
            tokens_post = tokenizer.convert_ids_to_tokens(input_ids_post[0])
            if return_options.get("tokens_post", False):
                result["tokens_post"] = tokens_post
        
        # Store input_ids if requested
        if return_options.get("input_ids", False):
            result["input_ids"] = input_ids
        
        # Append results
        results.append(result)

        # Handle printing logic if required
        if print_flag:
            if print_options.get("print_input_ids", False):
                print("Original token IDs:", input_ids)
            
            if print_options.get("print_tokens", False):
                print("Tokens:", tokens)
            
            if print_options.get("print_tokens_post", False):
                print("Tokens after re-encoding:", tokens_post)
            
            if print_options.get("print_decoded_text", False):
                print("Decoded text:", text)

            if print_options.get("print_lengths", False):
                print("Length of original token IDs:", len(input_ids))
                print("Length of re-encoded token IDs:", len(input_ids_post[0]))
                print("Number of words in decoded text:", len(text.split(' ')))

            print('\n-------------------\n')
        if max_rows is not None and i >= max_rows:
            break
    
    # Return results based on return options
    if len(return_options) == 1 and return_options.get("text", False):
        # Return only the text if that's all that's requested
        return [result["text"] for result in results]
    return results

# Example usage for actual script running
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tokenizer_path = os.path.join(current_dir, '..', 'data', 'tokenizer', 'tokenizer.json')
    data_file = os.path.join(current_dir, '..', 'processed_output', '00000_00000_shuffled.ds')

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(tokenizer_path)

    # Control what to print
    print_options = {
        "print_input_ids": False,
        "print_tokens": False,
        "print_tokens_post": False,
        "print_decoded_text": True,
        "print_lengths": False
    }

    # Process and print data
    process_data(data_file, tokenizer, print_options)