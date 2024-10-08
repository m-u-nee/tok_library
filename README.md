
---

# **Tokenization Pipeline**

## **Project Overview**

This project provides tools for preprocessing text datasets and tokenizing them using Hugging Face's `DocumentTokenizer`. It supports datasets in various formats (Hugging Face, JSONL, Parquet) and offers execution both locally and in distributed environments like Slurm.

Key components of the project:
- Data preprocessing and tokenization
- Tokenized data inspection
- Unit tests for validation

## **Project Structure**

- **`src/tok_library/`**: Main code for preprocessing and inspecting tokenized data.
  - **`data/`**: Contains tokenizer configuration files (e.g., `tokenizer.json`) and is the location for datasets (if used locally).
  - **`preprocess_data.py`**: Script for data preprocessing and tokenization.
  - **`inspect_binary.py`**: Script for inspecting and verifying tokenized data.
- **`tests/`**: Unit tests to ensure that tokenization and preprocessing work correctly.
  - **`test_tokenization_round_trip.py`**: Ensures that the tokenization process can round-trip (tokenize and decode back to original text) without errors.

Ensure the `src/tok_library/data/` folder contains the necessary tokenizer files, such as `tokenizer.json`.

----------------

## **Usage**

### 1. **Data Preprocessing** (`preprocess_data.py`)

This script handles the preprocessing and tokenization of datasets. It is fully configurable using a YAML configuration file (`config.yml`), which specifies the tokenizer, dataset format, output location, and execution options (local or Slurm).

#### **Example Usage:**

Create a `config.yml` file with all necessary parameters (e.g., tokenizer path, dataset path, Slurm settings) and run the script:

```bash
python preprocess_data.py
```

#### **Configuration Example (`config.yml`):**

```yaml
tokenizer_name_or_path: '../data/tokenizer/tokenizer.json'
dataset: '../data/my_dataset.parquet'
reader: 'parquet'
output_folder: '../processed_output/'
slurm: false
n_tasks: 4
shuffle: true
```

Key configuration options include:
- `tokenizer_name_or_path`: Path to the tokenizer.
- `dataset`: Path to the dataset file.
- `reader`: Dataset format (`hf`, `jsonl`, or `parquet`).
- `output_folder`: Folder to save processed data.
- Slurm options (e.g., `partition`, `n_tasks`, `time`, `mem_per_cpu_gb`).

---

### 2. **Inspecting Tokenized Data** (`inspect_binary.py`)

This script allows you to inspect tokenized data, decode it back to text, and verify its correctness. It supports configurable options for what to print and return.

#### **Example Usage:**

```bash
python inspect_binary.py
```

The script will load the tokenized data from the `.ds` file and print decoded text based on the options provided in the code.

---

### 3. **Unit Testing** (`test_tokenization_round_trip.py`)

This script performs round-trip tokenization and decoding, comparing the original text with the decoded text to verify consistency. It uses a YAML config file (`test_config.yml`) and logs errors or differences found during the test.

#### **Example Usage:**

Run the test suite with:

```bash
python -m unittest -v tests/test_tokenization_round_trip.py
```

The script will process a specified number of rows, log any inconsistencies, and (optionally) remove the processed data after the test is complete. Configure options like `num_rows_to_process` in the `USER_CONFIG` dictionary within the test file.

