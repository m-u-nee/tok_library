import os
import unittest
from argparse import Namespace
from src.inspect_binary import initialize_tokenizer, process_data
from src.preprocess_data import preprocess_data_main
import pandas as pd
import shutil
import difflib
import logging

# Configure logging
logging.basicConfig(filename='tests/test_log.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTokenizationRoundTrip(unittest.TestCase):
    def setUp(self):
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
            dataset='/Users/mattia/Desktop/pleias/tok_library/data/ItalianPD_1.parquet',
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

    def show_diff(self, text1, text2):
        diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
        diff_output = '\n'.join(diff)
        logger.error("Differences between original and decoded text:\n%s", diff_output)

    def clean_directory(self, directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                logger.error(f"Error removing {item_path}: {e}")

    def test_round_trip(self):
        # Step 1: Tokenize with preprocess_data_main
        preprocess_data_main(self.args)

        # Step 2: Convert the tokenized data back to text
        print_options = {
            "print_input_ids": False,
            "print_tokens": False,
            "print_tokens_post": False,
            "print_decoded_text": False,
            "print_lengths": False
        }
        
        results = process_data(self.processed_data_path, self.tokenizer, print_options)

        # Step 3: Compare each row of the parquet file with the decoded text
        df = pd.read_parquet(self.args.dataset)
        incorrect_rows = 0  # Track the number of incorrect rows
        total_rows = len(df)  # Total rows in the dataset
        logger.info(f"Number of rows in DataFrame: {len(df)}")
        logger.info(f"Number of results: {len(results)}")

        # Step 4: Compare each row and log differences without raising an error
        for idx, row in df.iterrows():
            original_text = row[self.args.column].strip()

            # Get the decoded text
            decoded_text = results[idx]['text']

            # Remove all special tokens from the decoded text
            for special_token in self.tokenizer.all_special_tokens:
                decoded_text = decoded_text.replace(special_token, "").strip()

            # Compare the original and decoded text
            if original_text != decoded_text:
                incorrect_rows += 1  # Increment the count of incorrect rows
                self.show_diff(original_text, decoded_text)  # Show the diff for the row

        # Step 5: Raise a single assertion based on the number of incorrect rows
        self.assertTrue(incorrect_rows == 0, f"{incorrect_rows}/{total_rows} rows did not match the expected text.")

        # Step 6: Clean up the processed data (optional)
        remove_processed_data = False
        if remove_processed_data:
            self.clean_directory(self.args.output_folder)
            self.clean_directory(self.args.logging_dir)

if __name__ == "__main__":
    unittest.main(buffer=True)