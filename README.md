# Bookish Couscous

### Installation

Local install.
```{bash}
conda create -n ouroboros python=3.10
conda activate ouroboros
pip install -e .
```

To install mamba dependencies:
```
pip install mamba-ssm
```

### Train

```python training.py --model_name_or_path <model_id> --train_file <path_jsonl> --output_dir <directory> --checkpointing_steps <int> --num_train_epochs 1 --learning_rate <lr> --chunk_size <int> --batch_size <int> --decoder_config <path_json> --encoder_config <path_json>```

### Evaluate
```python evaluate.py --base_model <model_id>  --config <path_json> --eval_file <path_jsonl> --chunk_size <int> --batch_size <int> --output_dir <directory> --ckpt_path <path_to_checkpoint>```

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
5. Set the path to `/home/<yourusername>/sudo-docker.sh`.
6. Open current project as folder in VSCode (command + O).
7. Press Command + Shift + P, and type + select `Dev Container: Open Current Folder in Container`. Press enter on root folder, choose cuda if asked.
8. Wait for building the environment and once that is done, you can continue development inside the container.


### Running using Docker Compose

* Training can be run as:
```bash
docker compose up -d train-autoencoder
```