Your `README` requires the following updates based on the code changes you provided:

1. **Update to `preprocess_data.py` description**: You now use a YAML config file to set parameters for `preprocess_data.py`. Also, you handle both local and Slurm-based execution with more configuration options.

2. **Update to `inspect_binary.py` description**: The tokenizer path and the data inspection process are configurable, with more detailed print options.

3. **Update to `test_tokenization_round_trip.py` description**: The unit test script is heavily reliant on YAML configuration, and now logs errors with more detail using a logger.

Here is the updated version of your `README`:

---

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

---

This README now reflects the changes to your code. If you have additional changes in the future, don't forget to revisit this document.