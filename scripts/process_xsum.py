import datasets


def prepare(x):
    inputs = f'Document:\n{x["document"]}\Summary:\n'
    label_str = x["summary"]
    return {
        "inputs": inputs,
        "label_str": label_str,
    }


dataset = datasets.load_dataset("EdinburghNLP/xsum")
dataset = dataset.map(
    prepare, batched=False, remove_columns=dataset["train"].column_names
)
print(dataset["train"][0])
dataset["train"].to_json("data/xsum/train.jsonl")
dataset["validation"].to_json("data/xsum/validation.jsonl")
dataset["test"].to_json("data/xsum/test.jsonl")
