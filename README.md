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
                                    --encoder <hf_model> \
                                    --decoder <hf_model> \
                                    --train_file <train_dataset.jsonl> \
                                    --output_dir <output_path> \
                                    --checkpointing_steps 1000 \
                                    --num_train_epochs 1 \
                                    --learning_rate 1e-5\
                                    --chunk_size <chunk_size> \
                                    --batch_size <batch_size> \
                                    --lr_scheduler_type constant
```


### Sequence-Level Evaluation
```
python evaluate.py --base_model <model_id>  --config <path_json> --eval_file <path_jsonl> --chunk_size <int> --batch_size <int> --output_dir <directory> --ckpt_path <path_to_checkpoint>
```

### Token-Level Evaluation

#### CoNNL-2003
Prepare ConNL test set to fixed size chunk length
```
python eval_analysis/connl/data_prep.py --input_path <connl_test.jsonl>\
                                        --output_path </path/to/connl256.jsonl>\
                                        --chunk_size 256
```

Reconstruct using decoder
```
python evaluate.py --base_model <model_id>  --config <path_json> --eval_file </path/to/connl256.jsonl> --chunk_size 256--batch_size <int> --output_dir <directory> --ckpt_path <path_to_checkpoint>
```

NER and Pos analysis
```
python eval_analysis/connl/ner_analysis.py --connl_reference </path/to/connl256.jsonl> \
                        --connl_generated </path/to/generated.jsonl> \
                        --output_path </path/to/analysis.csv>
```

# Paper
[Characterizing Mamba's Selective Memory using Auto-Encoders](https://arxiv.org/abs/2512.15653)

AACL 2025. Oral Presentation.

Tamanna Hossain, Robert L. Logan IV, Ganesh Jagadeesan, Sameer Singh, Joel Tetreault, Alejandro Jaimes