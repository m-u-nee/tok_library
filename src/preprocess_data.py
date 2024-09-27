import yaml
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader, ParquetReader

def load_config(config_file):
    """Loads the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data_main(config):
    # Sanity check for slurm
    if config['slurm'] and config['partition'] is None:
        raise Exception(
            "When slurm is specified, partition must also be provided.")
    elif not config['slurm'] and config['partition'] is not None:
        raise Exception(
            "partition can only be used when slurm is specified.")

    # Build datatrove reader
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
    config = load_config('config.yml')
    preprocess_data_main(config)