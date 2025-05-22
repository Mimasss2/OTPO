# OTPO 
This is the official implementation for our paper "Optimal Transport-Based Token Weighting scheme for Enhanced Preference Optimization". By emphasizing semantically meaningful token pairs and de-emphasizing less relevant ones, OTPO introduces a context-aware token weighting scheme that yields a more contrastive reward difference estimate. This adaptive weighting enhances reward stability, improves interpretability, and ensures that preference optimization focuses on meaningful differences between responses.

The main scripts for implementing OTPO lies in the `scripts` folder.

## Installation

To run the code of OTPO, first create a virtual environment using e.g. conda:
```bash
conda create -n otpo python=3.10
```

Activate the created environment, and install pytorch `v2.1.2` based in your hardware according to [Pytorch Installation Page](https://pytorch.org/get-started/previous-versions/), e.g.:
```bash
pip install torch==2.1.2
```

Install the rest packages:
```bash
pip install --upgrade pip setuptools
pip install -e . # Here we use an older version of alignment-handbook
pip install huggingface-hub==0.24.2
pip install accelerate==1.0.0
pip install flash-attn==2.5.5 --no-build-isolation
pip install wandb
pip install POT==0.9.4
echo -ne 'Y\n' | pip uninstall transformer_engine
```

Finally, log into your huggingface account by:
```bash
huggingface-cli login
```

[Optional] log into you wandb account by:
```bash
wandb login
```

## Model Training

We provide 4 training configuration files for the 4 training setups reported. The training config is set for 4xA100 GPUs for Llama-3-8B, and 2xA100 GPUs for Llama-3.2-3B.

- Llama-3-8B + UltraFeedback
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --deepspeed_multinode_launcher standard --use_deepspeed --num_processes <num_cards> scripts/run_otpo.py training_configs/otpo-llama3-8b-instruct-ultrafeedback.yaml
```

- Llama-3-8B + HelpSteer2
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --deepspeed_multinode_launcher standard --use_deepspeed --num_processes <num_cards> scripts/run_otpo.py training_configs/otpo-llama3-8b-instruct-helpsteer2.yaml
```

- Llama-3.2-3B + UltraFeedback
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --deepspeed_multinode_launcher standard --use_deepspeed --num_processes <num_cards> scripts/run_otpo.py training_configs/otpo-llama3_2-3b-instruct-ultrafeedback.yaml
```

- Llama-3.2-3B + HelpSteer2
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --deepspeed_multinode_launcher standard --use_deepspeed --num_processes <num_cards> scripts/run_otpo.py training_configs/otpo-llama3_2-3b-instruct-helpsteer2.yaml
```


## Evaluation

We follow the official implementation for evaluation on [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval), the rest general ability evaluations using [harness benchmark](https://github.com/EleutherAI/lm-evaluation-harness/tree/main). 

Specifically in the harness repository, we use the folloing task for each evaluation:
- MMLU: mmlu
- GSM8K: gsm8k
- ARC: arc_challenge
- HellaSwag: hellaswag
- PiQA: piqa
