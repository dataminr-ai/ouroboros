# Ouroboros

### Installation

Local install.
```{bash}
conda create -n ouroboros python=3.10
conda activate ouroboros
pip install -e .
```

Recommendation: If installing on a GPU machine, use the `fast` extra to install `mamba-ssm` and `causal-conv1d`.
(Which can speed up parts of training/inference).
```
pip install -e ".[fast]"
```

If you are planning on contributing to the project, please install the dev dependencies and set up the pre-commit checks.
```
pip install -e ".[dev]"
pre-commit install
```


### Train Decoder
Fixed sequence length:
```
python src/ouroboros/training.py  \
                                    --encoder state-spaces/mamba-130m-hf \
                                    --decoder state-spaces/mamba-130m-hf \
                                    --train_file train_subset_10k.jsonl \
                                    --output_dir models/reconstructor/fixed/4 \
                                    --checkpointing_steps 1000 \
                                    --num_train_epochs 1 \
                                    --learning_rate 1e-5\
                                    --chunk_size 4 \
                                    --batch_size 200 \
                                    --lr_scheduler_type constant
```

Mixed sequence length:
```
python src/ouroboros/training.py  \
                                    --encoder state-spaces/mamba-130m-hf \
                                    --decoder state-spaces/mamba-130m-hf \
                                    --train_file train_subset_10k.jsonl \
                                    --output_dir models/reconstructor/mixed \
                                    --checkpointing_steps 1000 \
                                    --num_train_epochs 1 \
                                    --learning_rate 1e-5\
                                    --batch_size 2 \
                                    --lr_scheduler_type constant \
                                    --mixed_chunk True
```

### Evaluate Decoder
```python evaluate.py --base_model <model_id>  --config <path_json> --eval_file <path_jsonl> --chunk_size <int> --batch_size <int> --output_dir <directory> --ckpt_path <path_to_checkpoint>```

### Train Cache

```
python src/ouroboros/training_cache.py  \
                                --decoder state-spaces/mamba-130m-hf \
                                --train_file piqa_formatted.jsonl \
                                --output_dir models/piqa \
                                --checkpointing_steps 100 \
                                --num_train_epochs 1 \
                                --learning_rate 1e-5\
                                --batch_size 10 \
                                --lr_scheduler_type constant \
```

Use regularization:
```
python src/ouroboros/training_cache.py  \
                                --decoder state-spaces/mamba-130m-hf \
                                --train_file piqa_formatted.jsonl \
                                --output_dir models/piqa \
                                --checkpointing_steps 100 \
                                --num_train_epochs 1 \
                                --learning_rate 1e-5\
                                --batch_size 10 \
                                --lr_scheduler_type constant \
                                --reg True \
                                --reg_strength 0.005 \
                                --reconstructor models/reconstructor/mixed
```

Initialize cache with a prompt:
```
python src/ouroboros/training_cache.py  \
                                --decoder state-spaces/mamba-130m-hf \
                                --train_file piqa_formatted.jsonl \
                                --output_dir models/piqa \
                                --checkpointing_steps 100 \
                                --num_train_epochs 1 \
                                --learning_rate 1e-5\
                                --batch_size 10 \
                                --lr_scheduler_type constant \
                                --starting_prompt "Pick the best option that answers the question.\n"
```
### Decode Cache
```
python src/ouroboros/decode_cache.py  \
                                --decoder models/reconstructor/mixed \
                                --learned_cache models/piqa/step_100/training_state.bin \
                                --tokenizer state-spaces/mamba-130m-hf
```

### Developing using [VSCode Devcontainers](https://code.visualstudio.com/docs/devcontainers/containers)

0. Install Dev Containers extension. Make sure your dockerfile is up-to-date.
1. If required, connect to the server via [VSCode Remote Development](https://code.visualstudio.com/docs/remote/remote-overview).
2. If you can run docker commands without sudo, jump directly to step 6. If not, in your home folder, create a file called `sudo-docker.sh` and the contents should be as follows:
    ```bash
    #!/bin/sh

    sudo docker "$@"
    ```
3. In VSCode task bar menus, select code > settings
4. In the search bar in the settings page, enter `docker path`. This setting should be visible in User tab as Dev > Containers: Docker Path (Applies to all profiles).
5. Set the path to `/home/<yourusername>/sudo-docker.sh` and make it executable (`chmod +x /home/<yourusername>/sudo-docker.sh`)
6. Open current project as folder in VSCode (command + O).
7. Press Command + Shift + P, and type + select `Dev Container: Open Current Folder in Container`. Press enter on root folder, choose cuda if asked.
8. Wait for building the environment and once that is done, you can continue development inside the container.


### Running using Docker Compose

* Training can be run as:
```bash
docker compose up -d train-autoencoder
```
