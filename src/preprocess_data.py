from datatrove.pipeline.tokens import DocumentTokenizer
import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader, ParquetReader


def get_args():
    # More arguments can be specified, have a look at https://github.com/huggingface/datatrove if you need details.
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer-name-or-path", type=str, required=True,
                        help="Path to tokenizer directory or model id on Hugging Face Hub.")
    parser.add_argument("--eos-token", type=str, default=None,
                        help="EOS token to add after each document. Default: None")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to the output folder for tokenized documents.")
    parser.add_argument("--logging-dir", type=str, default=None,
                        help="Path to folder for preprocessing logs. Default: None")
    parser.add_argument("--n-tasks", type=int, default=8,
                        help="Number of tasks for preprocessing. Default: 8")
    parser.add_argument("--n-workers", type=int, default=-1,
                        help="Number of workers. Default: -1 (no limit)")
    parser.add_argument("--shuffle", type=bool, default=False,
                        help="Shuffle the dataset. Default: False")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000,
                        help="Batch size for tokenization. Default: 1000")

    # Dataset arguments (combined for all types)
    parser.add_argument("--reader", type=str, required=True, choices=['hf', 'jsonl', 'parquet'],
                        help="Type of dataset to process: 'hf', 'jsonl', or 'parquet'")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset file/folder or Hugging Face hub repository")
    parser.add_argument("--column", type=str, default="text",
                        help="Column to preprocess. Default: text")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (for Hugging Face datasets). Default: train")
    parser.add_argument("--glob-pattern", type=str, default=None,
                        help="Glob pattern to filter files (for jsonl/parquet). Default: None")

    # Slurm-related arguments
    parser.add_argument("--slurm", action="store_true",
                        help="Use Slurm for job execution")
    parser.add_argument("--partition", type=str, default=None,
                        help="Slurm partition to use when --slurm is specified")
    parser.add_argument("--qos", type=str, default=None,
                        help="Quality of Service (QOS) for Slurm job. Optional, cluster-specific. Default: None")
    parser.add_argument("--time", type=str, default="20:00:00",
                        help="Maximal time for a job. Default: 20:00:00")
    parser.add_argument("--email", type=str, default=None,
                        help="Get an email notification when the job is finished. Default: None")
    parser.add_argument("--cpus-per-task", type=int, default=1,
                        help="How many CPUs to give each task. Default: 1")
    parser.add_argument("--mem-per-cpu-gb", type=int, default=2,
                        help="Memory per CPU in GB. Default: 2")

    return parser.parse_args()


def preprocess_data_main(args):
    # Sanity check for slurm
    if args.slurm and args.partition is None:
        raise Exception(
            "When --slurm is specified, --partition must also be provided.")
    elif not args.slurm and args.partition is not None:
        raise Exception(
            "--partition can only be used when --slurm is specified.")

    # Build datatrove reader
    if args.reader == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options={"split": args.split},
        )
    elif args.reader == "jsonl":
        datatrove_reader = JsonlReader(
            data_folder=args.dataset,
            glob_pattern=args.glob_pattern,
            text_key=args.column,
        )

    elif args.reader == "parquet":
        datatrove_reader = ParquetReader(
            data_folder=args.dataset,
            glob_pattern=args.glob_pattern,
            text_key=args.column,
        )
    else:
        raise Exception(
            f"args.reader defined as {args.reader}, must be in [hf, jsonl, parquet]")

    if args.slurm:
        # launch a job in a slurm cluster
        preprocess_executor = SlurmPipelineExecutor(
            job_name="tokenization",
            pipeline=[
                datatrove_reader,
                DocumentTokenizer(
                    output_folder=args.output_folder,
                    tokenizer_name_or_path=args.tokenizer_name_or_path,
                    eos_token=args.eos_token,
                    shuffle=args.shuffle,
                    max_tokens_per_file=3e8,
                ),
            ],
            partition=args.partition,
            time=args.time,
            tasks=args.n_tasks,
            logging_dir=args.logging_dir,
            workers=args.n_workers,
            cpus_per_task=args.cpus_per_task,
            qos=args.qos,
            mail_user=args.email,
            mem_per_cpu_gb=args.mem_per_cpu_gb,
        )
    else:
        # run in interactive node
        preprocess_executor = LocalPipelineExecutor(
            pipeline=[
                datatrove_reader,
                DocumentTokenizer(
                    output_folder=args.output_folder,
                    tokenizer_name_or_path=args.tokenizer_name_or_path,
                    eos_token=args.eos_token,
                    shuffle=args.shuffle,
                    max_tokens_per_file=3e8,
                    batch_size=args.tokenizer_batch_size,
                ),
            ],
            tasks=args.n_tasks,
            logging_dir=args.logging_dir,
            workers=args.n_workers,
        )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    preprocess_main(_args)