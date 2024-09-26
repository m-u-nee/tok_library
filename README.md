# **Tokenization Pipeline**

## **Project Overview**

This project provides tools for preprocessing text datasets and tokenizing them using Hugging Face's `DocumentTokenizer`. It supports datasets in various formats (Hugging Face, JSONL, Parquet) and offers execution both locally and in distributed environments like Slurm.

Key components of the project:
- Data preprocessing and tokenization
- Tokenized data inspection
- Unit tests for validation

## **Project Structure**

- **data/**: Contains dataset files and the tokenizer configuration (e.g., `tokenizer.json`).
- **src/**: Main scripts for preprocessing and inspecting tokenized data.
- **tests/**: Unit tests to ensure that tokenization works correctly.

Ensure the `data/tokenizer` folder contains the necessary tokenizer files, such as `tokenizer.json`.

---

## **Usage**

### 1. **Data Preprocessing** (`preprocess_data.py`)

This script handles the preprocessing and tokenization of datasets. You can configure the tokenizer, dataset format, and execution settings.

#### **Example Command:**

To run the preprocessing script, specify the tokenizer path, dataset, output folder, and other options such as the dataset format and the column to process.

#### **Key Arguments:**

- `--tokenizer-name-or-path`: Path to the tokenizer or Hugging Face model ID.
- `--dataset`: Path to the dataset file or Hugging Face repository.
- `--reader`: Dataset format (`hf`, `jsonl`, or `parquet`).
- `--output-folder`: Directory for storing tokenized data.

Additional arguments like `--slurm`, `--n-tasks`, and `--shuffle` can be used for customization.

---

### 2. **Inspecting Tokenized Data** (`inspect_binary.py`)

This script is used to decode and inspect tokenized data, allowing you to verify that the tokenization process is correct.

To run the inspection, provide the path to the tokenized data and the tokenizer. The script will print the decoded text and other details based on specified options.

---

### 3. **Unit Testing** (`test_tokenization_round_trip.py`)

This script contains unit tests for verifying that tokenization works as expected. The tests perform a "round-trip" by tokenizing and then decoding the text to compare it with the original.

To run the tests, use Python's `unittest` module.

---

