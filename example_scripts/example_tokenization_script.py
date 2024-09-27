from tok_library.preprocess_data import preprocess_data_main
import os
import yaml
print(f"Current working directory: {os.getcwd()}")
config_path = "example_scripts/example_tokenization_config.yml"
# Extract the tokenizer_path_or_name from the config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    tokenizer_path_or_name = config["tokenizer_name_or_path"]
print(f"Tokenizer path or name: {tokenizer_path_or_name}")
preprocess_data_main(config_path)

print("Tokenization complete.")