# Bookish Couscous

### Encoding Data
```python encode_dataset.py --input_file <filename.jsonl> --model_id <model_id> --chunk_size <int> --batch_size <int> --output_file <output_files_dir> --checkpoints <int_for_sharding>```

### Training

```python training.py --model_name_or_path <model_id> --config_path <path_config.jsonl> --train_path <training_files_dir> --data_shard_len <int> --output_dir <model_output_dir> --checkpointing_steps <int> --num_train_epochs <int> --learning_rate <lr>```

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