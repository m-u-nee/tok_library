import os
import unittest
import yaml
from tok_library.inspect_binary import load_tokenizer, process_tokenized_data  # Updated path to reflect the project structure
from tok_library.preprocess_data import preprocess_data_main  # Updated path
import pandas as pd
import shutil
import difflib
import logging

"""
This script tests the round-trip tokenization process. It tokenizes a dataset, decodes the tokenized data back to text,
and compares the original text with the decoded text. The test passes if all rows match.
run with python -m unittest -v tests/test_tokenization_round_trip.py
"""

# USER CONFIGURATION
USER_CONFIG = {
    "config_file_path": '/Users/mattia/Desktop/pleias/tok_library/tests/test_config.yml',  # File path to the config file for preprocess_data_main.
    "log_file_path": 'tests/test_log.log',  # Logs information during test, mostly errors if any occur for debugging
    "num_rows_to_process": float("inf"),  # Number of rows to check. Set low for faster testing.
    "print_options": {
        "print_input_ids": False,
        "print_tokens": False,
        "print_tokens_post": False,
        "print_decoded_text": False,
        "print_lengths": False
    },  # Options for printing the decoded results. Set to False to disable printing.
    "remove_processed_data_after_test": True  # Remove the processed data after the test is completed.
}


def setup_logger(log_file_path):
    """Configures and clears the log file."""
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Clear the log file
    open(log_file_path, 'w').close()
    return logger

def load_config(config_file):
    """Loads the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class TestTokenizationRoundTrip(unittest.TestCase):
    """Unit test for tokenization round-trip consistency."""

    @classmethod
    def setUpClass(cls):
        """Set up the logger for the entire test class."""
        cls.logger = setup_logger(USER_CONFIG['log_file_path'])

    def setUp(self):
        """Set up test variables and paths."""
        print("Setup is running")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the configuration from the file
        config_file_path = USER_CONFIG['config_file_path']
        self.config = load_config(config_file_path)

        # Set tokenizer and paths from the config
        print(f"Tokenizer path: {self.config['tokenizer_name_or_path']}")
        self.tokenizer = load_tokenizer(self.config['tokenizer_name_or_path'])
        print("Tokenizer initialized")
        self.processed_data_path = os.path.join(self.config['output_folder'], "00000_unshuffled.ds")

    def show_diff(self, text1, text2):
        """Log the differences between two texts in a human-readable format. This is called when a test fails."""
        diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
        diff_output = '\n'.join(diff)
        self.logger.error("Differences between original and decoded text:\n%s", diff_output)

    def clean_directory(self, directory):
        """Remove files and directories inside a given directory."""
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            self.logger.info(f"Successfully cleaned directory: {directory}")
        except Exception as e:
            self.logger.exception(f"Failed to clean directory {directory}: {e}")
    
    def clean_up_after_test(self):
        """Clean up temporary test files. Comment out if you want to inspect the files."""
        self.clean_directory(self.config['output_folder'])
        self.clean_directory(self.config['logging_dir'])

    
    def test_round_trip(self):
        """Test round-trip tokenization: original -> tokenized -> decoded."""
        # Step 1: Tokenize with preprocess_data_main
        preprocess_data_main(USER_CONFIG['config_file_path'])
        print("Tokenization complete")

        # Step 2: Decode the tokenized data back to text
        print_options = USER_CONFIG['print_options']
        num_rows_to_process = USER_CONFIG['num_rows_to_process']

        # Process the data and decode back to text
        results = process_tokenized_data(
            self.processed_data_path,
            self.tokenizer,
            print_options,
            return_options={"text": True},
            max_rows=num_rows_to_process
        )

        # Step 3: Load the original dataset
        df = pd.read_parquet(self.config['dataset'])
        incorrect_rows = 0
        total_rows = min(len(df), num_rows_to_process)

        self.logger.info(f"Number of rows in DataFrame: {len(df)}")
        self.logger.info(f"Number of results: {len(results)}")

        # Step 4: Compare original and decoded text
        for idx, row in df.iterrows():
            if idx >= num_rows_to_process:
                break
            original_text = row[self.config['column']].strip()
            decoded_text = results[idx].strip()

            # Remove special tokens from the decoded text
            for special_token in self.tokenizer.all_special_tokens:
                decoded_text = decoded_text.replace(special_token, "").strip()

            if original_text != decoded_text:
                incorrect_rows += 1
                self.logger.error(f"Row {idx} did not match the expected text.")
                self.show_diff(original_text, decoded_text)

        self.logger.info(f"Number of incorrect rows: {incorrect_rows} out of {total_rows} rows.")

        # Step 5: Assert that all rows match
        try:
            self.assertTrue(incorrect_rows == 0, f"{incorrect_rows}/{total_rows} rows did not match the expected text.")

        # Step 6: Clean up processed data
        finally:
            if USER_CONFIG['remove_processed_data_after_test']:
                self.logger.info("Cleaning up processed data.")
                self.clean_up_after_test()

    
if __name__ == "__main__":
    unittest.main(buffer=True)