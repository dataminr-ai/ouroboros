import datasets


def prepare(x):
    inputs = f'Question:\n{x["goal"]}\nA. {x["sol1"]}\nB. {x["sol2"]}\nAnswer: '
    label_str = "AB"[x["label"]]
    correct_sol = [x["sol1"], x["sol2"]][x["label"]]
    incorrect_sol = [x["sol1"], x["sol2"]][1 - x["label"]]
    positive = f'Question: {x["goal"]}\nAnswer: {correct_sol}'
    negative = f'Question: {x["goal"]}\nAnswer: {incorrect_sol}'
    return {
        "inputs": inputs,
        "label_str": label_str,
        "positive": positive,
        "negative": negative,
    }


dataset = datasets.load_dataset("ybisk/piqa", name="plain_text")
dataset = dataset.filter(lambda x: x["label"] in [0, 1])
dataset = dataset.map(
    prepare, batched=False, remove_columns=dataset["train"].column_names
)
print(dataset["train"][0])
dataset["train"].to_json("data/piqa/train.jsonl")
dataset["validation"].to_json("data/piqa/validation.jsonl")
dataset["test"].to_json("data/piqa/test.jsonl")
