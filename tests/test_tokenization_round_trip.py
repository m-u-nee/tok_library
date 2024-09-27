import os
import unittest
import yaml
from src.inspect_binary import initialize_tokenizer, process_data
from src.preprocess_data import preprocess_data_main
import pandas as pd
import shutil
import difflib
import logging




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
        cls.log_file_path = 'tests/test_log.log'
        cls.logger = setup_logger(cls.log_file_path)

    def setUp(self):
        """Set up test variables and paths."""
        print("Setup is running")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the configuration from the file
        config_file_path = os.path.join(current_dir, 'test_config.yml')  # Change this to your actual config path
        self.config = load_config(config_file_path)

        # Set tokenizer and paths from the config
        print(f"Tokenizer path: {self.config['tokenizer_name_or_path']}")
        self.tokenizer = initialize_tokenizer(self.config['tokenizer_name_or_path'])
        print(f"Tokenizer initialized")
        self.processed_data_path = os.path.join(self.config['output_folder'], "00000_unshuffled.ds")

    def show_diff(self, text1, text2):
        """Log the differences between two texts in a human-readable format. This is called when a test fails."""
        diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
        diff_output = '\n'.join(diff)
        self.logger.error("Differences between original and decoded text:\n%s", diff_output)

    def clean_directory(self, directory):
        """Remove files and directories inside a given directory. This is used to clean up the processed data 
        after the test is completed."""
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                self.logger.error(f"Error removing {item_path}: {e}")

    def test_round_trip(self):
        """Test round-trip tokenization: original -> tokenized -> decoded."""
        # Step 1: Tokenize with preprocess_data_main
        preprocess_data_main('/Users/mattia/Desktop/pleias/tok_library/tests/test_config.yml')  # Pass the config file path
        print("Tokenization complete")

        # Step 2: Decode the tokenized data back to text
        print_options = {
            "print_input_ids": False,
            "print_tokens": False,
            "print_tokens_post": False,
            "print_decoded_text": False,
            "print_lengths": False
        }
        num_rows_to_process = float("inf")

        # Process the data and decode back to text
        results = process_data(
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
            remove_processed_data_after_test = True
            if remove_processed_data_after_test:
                self.logger.info("Cleaning up processed data.")
                self.clean_up_after_test()

    def clean_up_after_test(self):
        """Clean up temporary test files. Comment out if you want to inspect the files."""
        self.clean_directory(self.config['output_folder'])
        self.clean_directory(self.config['logging_dir'])

if __name__ == "__main__":
    unittest.main(buffer=True)