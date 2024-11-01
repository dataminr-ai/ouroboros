import datasets


def prepare(x):
    inputs = f'Article:\n{x["article"]}\Summary:\n'
    label_str = f'Summary:\n{x["highlights"]}'
    return {
        "inputs": inputs,
        "label_str": label_str,
    }


dataset = datasets.load_dataset("abisee/cnn_dailymail", name="plain_text")
dataset = dataset.map(
    prepare, batched=False, remove_columns=dataset["train"].column_names
)
print(dataset["train"][0])
dataset["train"].to_json("data/cnn_daily_mail/train.jsonl")
dataset["validation"].to_json("data/cnn_daily_mail/validation.jsonl")
dataset["test"].to_json("data/cnn_daily_mail/test.jsonl")
