import unittest
from argparse import Namespace
from src.inspect_binary import initialize_tokenizer, process_data
from src.preprocess_data import preprocess_data_main
import pandas as pd


class TestTokenizationRoundTrip(unittest.TestCase):
    def setUp(self):
        # Manually setting up the args that would normally come from the command line
        self.args = Namespace(
            tokenizer_name_or_path='/Users/mattia/Desktop/pleias/rag_data_exploration/tokenizer/tokenizer.json',
            eos_token="<|end_of_text|>",
            output_folder='/Users/mattia/Desktop/pleias/tok_library/tests/test_processed_output',
            logging_dir='/Users/mattia/Desktop/pleias/tok_library/tests/test_processed_output',
            n_tasks=1,
            n_workers=-1,
            shuffle=False,
            tokenizer_batch_size=100,
            reader="parquet",  # Simulating a Hugging Face reader
            dataset="/Users/mattia/Desktop/pleias/tok_library/tests/test_data/random_row.parquet",
            column="text",
            split="train",
            glob_pattern=None,
            slurm=False,
            partition=None,
            qos=None,
            time="20:00:00",
            email='mattia.u.nee@gmail.com',
            cpus_per_task=1,
            mem_per_cpu_gb=2
        )

        # Initialize tokenizer
        self.tokenizer_path = '/Users/mattia/Desktop/pleias/rag_data_exploration/tokenizer'
        self.tokenizer = initialize_tokenizer(self.tokenizer_path)

        # File for tokenized data
        self.data_file = '/Users/mattia/Desktop/pleias/rag_data_exploration/processed_output/00000_00000_shuffled.ds'

    def test_round_trip(self):
        # Step 1: Tokenize with preprocess_data_main
        # Runs the tokenization process with the args
        preprocess_data_main(self.args)

        # Step 2: Convert the tokenized data back to text
        print_options = {
            "print_input_ids": False,
            "print_tokens": False,
            "print_tokens_post": False,
            "print_decoded_text": True,
            "print_lengths": True
        }
        tokenizer = initialize_tokenizer(self.tokenizer_path)
        processed_data_path = self.args.output_folder + "/00000_unshuffled.ds"
        results = process_data(processed_data_path, tokenizer, print_options)

        # Step 3: Compare the original text with the decoded text. The original text is the first row of the parquet file, under column 'text'
        
        # Read the first row of the parquet file
        df = pd.read_parquet(self.args.dataset)
        original_text = df[self.args.column].iloc[0].strip()

        # Get the decoded text and remove the eos_token
        decoded_text = results[0]['text'].strip()
        eos_token = self.args.eos_token
        if decoded_text.endswith(eos_token):
            decoded_text = decoded_text[:-len(eos_token)].strip()

        # Step 4: Compare the stripped original and decoded text
        self.assertEqual(original_text, decoded_text)


if __name__ == "__main__":
    unittest.main()
