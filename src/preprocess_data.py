import yaml
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader, ParquetReader

"""
To use, create a config.yml file with format specified below,
"""

def load_config(config_file):
    """Loads the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_datatrove_reader(config):
    """
    Creates a DataTrove reader based on the specified configuration.

    Args:
        config (dict): A dictionary containing configuration parameters for the reader.
            Expected keys include:
            - 'reader' (str): The type of reader ('hf', 'jsonl', or 'parquet').
            - 'dataset' (str): The path to the dataset.
            - 'glob_pattern' (str, optional): A glob pattern to filter files.
            - 'text_key' (str): The key in the dataset that contains the text data.

    Returns:
        datatrove_reader: An instance of the appropriate DataTrove reader.

    Raises:
        Exception: If the specified reader is not recognized."""
    
    if config['reader'] == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=config['dataset'],
            text_key=config['column'],
            dataset_options={"split": config['split']},
        )
    elif config['reader'] == "jsonl":
        datatrove_reader = JsonlReader(
            data_folder=config['dataset'],
            glob_pattern=config.get('glob_pattern'),
            text_key=config['column'],
        )
    elif config['reader'] == "parquet":
        datatrove_reader = ParquetReader(
            data_folder=config['dataset'],
            glob_pattern=config.get('glob_pattern'),
            text_key=config['column'],
        )
    else:
        raise Exception(
            f"config['reader'] defined as {config['reader']}, must be in [hf, jsonl, parquet]")
    return datatrove_reader

def preprocess_data_main(config_path):
    """
    Main function to preprocess data using DataTrove. Will save data in .ds format in specified output folder, along with some statistics.

    This function sets up and executes the data preprocessing pipeline based on the provided configuration.
    It supports both local execution and distributed execution on a Slurm cluster.

    Args:
        config (path): The path to the configuration file, expected to be in YAML format. Find example in example_preprocess_data_config.yml.
            Expected keys include:
            - 'slurm' (bool): Whether to use Slurm for distributed execution.
            - 'partition' (str): The Slurm partition to use (required if 'slurm' is True).
            - 'output_folder' (str): The folder to save preprocessed data.
            - 'tokenizer_name_or_path' (str): The name or path of the tokenizer to use.
            - 'eos_token' (str, optional): The end-of-sequence token.
            - 'shuffle' (bool): Whether to shuffle the data.
            - 'time' (str): The time limit for Slurm jobs.
            - 'n_tasks' (int): The number of tasks to run.
            - 'logging_dir' (str): The directory for logging.
            - 'n_workers' (int): The number of workers to use.
            - 'cpus_per_task' (int): The number of CPUs per task.
            - 'qos' (str, optional): Quality of Service for Slurm.
            - 'email' (str, optional): Email for Slurm notifications.
            - 'mem_per_cpu_gb' (float): Memory per CPU in GB.
            - 'tokenizer_batch_size' (int): Batch size for tokenization (local execution only).

    Raises:
        Exception: If Slurm configuration is inconsistent or invalid.

    Returns:
        None
    """
    config = load_config(config_path)
    # Sanity check for slurm
    if config['slurm'] and config['partition'] is None:
        raise Exception(
            "When slurm is specified, partition must also be provided.")
    elif not config['slurm'] and config['partition'] is not None:
        raise Exception(
            "partition can only be used when slurm is specified.")

    # Build datatrove reader
    datatrove_reader = create_datatrove_reader(config)
    
    if config['slurm']:
        # launch a job in a slurm cluster
        preprocess_executor = SlurmPipelineExecutor(
            job_name="tokenization",
            pipeline=[
                datatrove_reader,
                DocumentTokenizer(
                    output_folder=config['output_folder'],
                    tokenizer_name_or_path=config['tokenizer_name_or_path'],
                    eos_token=config.get('eos_token'),
                    shuffle=config['shuffle'],
                    max_tokens_per_file=3e8,
                ),
            ],
            partition=config['partition'],
            time=config['time'],
            tasks=config['n_tasks'],
            logging_dir=config['logging_dir'],
            workers=config['n_workers'],
            cpus_per_task=config['cpus_per_task'],
            qos=config.get('qos'),
            mail_user=config.get('email'),
            mem_per_cpu_gb=config['mem_per_cpu_gb'],
        )
    else:
        # run in interactive node
        preprocess_executor = LocalPipelineExecutor(
            pipeline=[
                datatrove_reader,
                DocumentTokenizer(
                    output_folder=config['output_folder'],
                    tokenizer_name_or_path=config['tokenizer_name_or_path'],
                    eos_token=config.get('eos_token'),
                    shuffle=config['shuffle'],
                    max_tokens_per_file=3e8,
                    batch_size=config['tokenizer_batch_size'],
                ),
            ],
            tasks=config['n_tasks'],
            logging_dir=config['logging_dir'],
            workers=config['n_workers'],
        )
    preprocess_executor.run()

if __name__ == "__main__":
    # Load configuration from a YAML file
    config_file_path = "preprocess_data_config.yml"
    preprocess_data_main(config_file_path)