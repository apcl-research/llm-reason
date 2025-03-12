# QLoRA for CodeLlama

This repository is a fork of the [poludmik/qlora_for_codegen](https://github.com/poludmik/qlora_for_codegen) repository.  This is a development repository where we conduct experiments using code LLMs.

Create a new virtualenv with packages in requirements.txt.  Installed CUDA version is 12.2.  Machine has 256GB system memory and four A5000 GPUs.  Fine-tuning the CodeLlama Instruct 70B model is possible using all four GPUs.

Model files are currently on nublar in a shared directory.  Create a symlink in this folder:

`
ln -s /nublar/models checkpoints
`

You can get different models by using the download.py script in the [Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt) repository.  That script downloads from Huggingface.

Example command to fine-tune:

`
$ python3 qlora.py --model_name_or_path checkpoints/codellama/CodeLlama-70b-Instruct-hf/ --ddp_find_unused_parameters False --bf16
`

The code in this repository will not modify the files in the checkpoints directories.  It will create an adapter LoRA model and store it in a folder called `output`.  Use caution since by default the scripts will store a checkpoint every 90 steps, and each checkpoint will be 11GB for the 70B model.

