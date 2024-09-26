import os
import unittest
from argparse import Namespace
from src.inspect_binary import initialize_tokenizer, process_data
from src.preprocess_data import preprocess_data_main
import pandas as pd
import glob
import shutil

""" run with python -m unittest tests/test_tokenization_round_trip.py """
"""This test is for the round trip of tokenization. It tokenizes the data, then decodes it back to text and compares it with the original text."""

class TestTokenizationRoundTrip(unittest.TestCase):
    def setUp(self):
        # Manually setting up the args that would normally come from the command line
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.args = Namespace(
            tokenizer_name_or_path=os.path.join(current_dir, '..', 'data', 'tokenizer', 'tokenizer.json'),
            eos_token="<|end_of_text|>",
            output_folder=os.path.join(current_dir, 'test_processed_output'),
            logging_dir=os.path.join(current_dir, 'test_processed_output'),
            n_tasks=1,
            n_workers=-1,
            shuffle=False,
            tokenizer_batch_size=100,
            reader="parquet", 
            dataset=os.path.join(current_dir, 'test_data', 'random_row.parquet'),
            column="text",
            split="train",
            glob_pattern=None,
            slurm=False,
            partition=None,
            qos=None,
            time="20:00:00",
            email=None,
            cpus_per_task=1,
            mem_per_cpu_gb=2
        )

        self.tokenizer_path = self.args.tokenizer_name_or_path
        self.tokenizer = initialize_tokenizer(self.tokenizer_path)
        self.processed_data_path = os.path.join(self.args.output_folder, "00000_unshuffled.ds")




    def clean_directory(self, directory):
        # Get all subdirectories and files in the directory and remove them
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isdir(item_path):
                    # Recursively delete the folder and all its contents
                    shutil.rmtree(item_path)
                else:
                    # Remove the file
                    os.remove(item_path)
            except Exception as e:
                print(f"Error removing {item_path}: {e}")


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
        
        results = process_data(self.processed_data_path, self.tokenizer, print_options)

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

        # Step 5: Clean up the processed data, optional
        remove_processed_data = True
        if remove_processed_data:
            self.clean_directory(self.args.output_folder)
            self.clean_directory(self.args.logging_dir)

if __name__ == "__main__":
    unittest.main()
