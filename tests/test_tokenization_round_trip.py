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
# Clear log file
open('tests/test_log.log', 'w').close()

class TestTokenizationRoundTrip(unittest.TestCase):
    def setUp(self):
        print("Setup is running")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.args = Namespace(
            tokenizer_name_or_path=os.path.join(current_dir, '..', 'data', 'tokenizer', 'tokenizer.json'),
            eos_token="<|end_of_text|>",
            output_folder=os.path.join(current_dir, 'test_processed_data'),
            logging_dir=os.path.join(current_dir, 'test_processed_data'),
            n_tasks=1,
            n_workers=-1,
            shuffle=False,
            tokenizer_batch_size=10,
            reader="parquet", 
            dataset='/Users/mattia/Desktop/pleias/tok_library/data/YoutubeCommons_1.parquet',
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
        print("Tokenization complete")
        # Step 2: Convert the tokenized data back to text
        print_options = {
            "print_input_ids": False,
            "print_tokens": False,
            "print_tokens_post": False,
            "print_decoded_text": False,
            "print_lengths": False
        }
        num_rows_to_process = float("inf")
        # Ensure that process_data limits the number of rows processed
        results = process_data(
            self.processed_data_path, 
            self.tokenizer, 
            print_options, 
            return_options={"text": True},
            max_rows=num_rows_to_process  # Add this parameter to limit rows in process_data
        )

        # Step 3: Load the parquet file and compare the rows with the decoded text
        df = pd.read_parquet(self.args.dataset)
        incorrect_rows = 0  # Track the number of incorrect rows
        total_rows = min(len(df), num_rows_to_process)  # Total rows to be processed
        logger.info(f"Number of rows in DataFrame: {len(df)}")
        logger.info(f"Number of results: {len(results)}")

        # Step 4: Compare each row and log differences without raising an error
        for idx, row in df.iterrows():
            # Break if we've reached the maximum number of rows to process
            if idx >= num_rows_to_process:
                break
            
            original_text = row[self.args.column]
            
            # Get the decoded text from results
            decoded_text = results[idx]

            # We remove trailing white spaces from both texts
            original_text = original_text.strip()
            decoded_text = decoded_text.strip()


            # Remove all special tokens from the decoded text
            for special_token in self.tokenizer.all_special_tokens:
                decoded_text = decoded_text.replace(special_token, "").strip()

            # Compare the original and decoded text
            if original_text != decoded_text:
                incorrect_rows += 1  # Increment the count of incorrect rows
                logger.error(f"Row {idx} did not match the expected text.")
                self.show_diff(original_text, decoded_text)  # Show the diff for the row
        logger.info(f"Number of incorrect rows: {incorrect_rows} out of {total_rows} rows.")
        # Step 5: Raise a single assertion based on the number of incorrect rows
        try:
            self.assertTrue(incorrect_rows == 0, f"{incorrect_rows}/{total_rows} rows did not match the expected text.")
            
        
        # Step 6: Clean up the processed data (optional)
        finally:
            logger.info("Cleaning up processed data.")
            remove_processed_data = True
            if remove_processed_data:
                self.clean_directory(self.args.output_folder)
                self.clean_directory(self.args.logging_dir)

if __name__ == "__main__":
    unittest.main(buffer=True)