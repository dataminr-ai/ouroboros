import argparse
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
PAD_LABEL = -100  
PAD_TOKEN = tokenizer.pad_token if tokenizer.pad_token is not None else "[PAD]" 


def subword_tokenize_with_labels_grouped(
    example: Dict[str, Any]
) -> List[List[Dict[str, Any]]]:
    """
    Convert a single CoNLL example into a list of tokens, 
    where each token is itself a list of subword dicts.

    Example input `example`:
      {
          "id": "0",
          "tokens": [...],
          "pos_tags": [...],
          "chunk_tags": [...],
          "ner_tags": [...]
      }

    We'll return :
      [
        [  # subwords for token 0
          { "subword": "S",  "pos_subword": 22,  ... },
          { "subword": "oc", "pos_subword": 22,  ... },
          ...
        ],
        [  # subwords for token 1
          ...
        ],
        ...
      ]

    So we don't split a single token's subwords across chunk boundaries unintentionally.
    """
    tokens = example["tokens"]
    pos_tags = example["pos_tags"]
    chunk_tags = example["chunk_tags"]
    ner_tags = example["ner_tags"]
    
    all_tokens_subwords = []

    for tok, pos, chk, ner in zip(tokens, pos_tags, chunk_tags, ner_tags):
        subwords = tokenizer.tokenize(tok, add_special_tokens=False)
        
        # Build the subword dicts for this single token
        subword_list = []
        for sw in subwords:
            subword_list.append({
                "subword": sw,
                "pos_subword": pos,
                "chunk_subword": chk,
                "ner_subword": ner,
                "orig_token": tok,
                "orig_pos": pos,
                "orig_chunk": chk,
                "orig_ner": ner
            })
        
        all_tokens_subwords.append(subword_list)
    
    return all_tokens_subwords


def create_pad_subword():
    """
    Create a subword dictionary representing a PAD token.
    Use PAD_LABEL for the labels, so the model can ignore them.
    """
    return {
        "subword": PAD_TOKEN,
        "pos_subword": PAD_LABEL,
        "chunk_subword": PAD_LABEL,
        "ner_subword": PAD_LABEL,
        "orig_token": "[PAD]",
        "orig_pos": PAD_LABEL,
        "orig_chunk": PAD_LABEL,
        "orig_ner": PAD_LABEL
    }


def build_fixed_size_chunks(
    conll_data: List[Dict[str, Any]],
    max_length: int = 256
) -> List[List[Dict[str, Any]]]:
    """
    Builds fixed-size chunks (exactly `max_length` subwords per chunk),
    without splitting any original token across chunk boundaries.

    Steps:
      1. For each CoNLL example, get a list of token-subword groups.
      2. Accumulate these as units. If adding the next token-subword group
         would exceed `max_length`, pad the current chunk to `max_length` with [PAD] tokens
         and start a new chunk.
      3. If at the end, the current chunk isn't full, pad it too.
    
    Returns a list of chunks, each chunk is a list of subword dicts, length == max_length.
    """
    all_chunks = []
    current_chunk = []
    
    for example in conll_data:
        token_subwords_list = subword_tokenize_with_labels_grouped(example)
        
        for subword_group in token_subwords_list:
            group_size = len(subword_group)

            # Handle edge case: a single token that yields more subwords than max_length
            if group_size > max_length:
                print(f"WARNING: A single token is split into {group_size} subwords "
                      f"which exceeds max_length={max_length}. Skipping token.")
                continue

            # If adding this group would exceed max_length, 
            # pad current_chunk to exactly max_length, push it, start a new one
            if len(current_chunk) + group_size > max_length:
                # pad the current chunk
                while len(current_chunk) < max_length:
                    current_chunk.append(create_pad_subword())
                
                all_chunks.append(current_chunk)
                current_chunk = []
            
            # Now add this token's subwords
            current_chunk.extend(subword_group)

            # If it exactly hits max_length, finalize it and start a new one
            if len(current_chunk) == max_length:
                all_chunks.append(current_chunk)
                current_chunk = []
    
    # After processing all tokens from all examples,
    # if there's a remainder that's not full, pad it and keep it
    if len(current_chunk) > 0:
        while len(current_chunk) < max_length:
            current_chunk.append(create_pad_subword())
        all_chunks.append(current_chunk)
    
    return all_chunks


def convert_chunks_to_conll_format(chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert the final chunks into a list of dicts. Each chunk is `max_length` subwords.
    
    We keep:
     - subword-level tokens/labels,
     - a de-duplicated list of the original tokens/labels,
     - a "text" field made by joining the de-duplicated original tokens.
    """
    new_data = []
    
    for i, chunk in enumerate(chunks):
        # --------------------------------------------------
        # 1) Gather subword fields
        # --------------------------------------------------
        subwords = [entry["subword"] for entry in chunk]
        pos_subwords = [entry["pos_subword"] for entry in chunk]
        chunk_subwords = [entry["chunk_subword"] for entry in chunk]
        ner_subwords = [entry["ner_subword"] for entry in chunk]
        
        # --------------------------------------------------
        # 2) Build a de-duplicated list of original tokens
        #    so each token appears only once per occurrence.
        # --------------------------------------------------
        dedup_orig_tokens = []
        for j, entry in enumerate(chunk):
            token_j = entry["orig_token"]
            pos_j = entry["orig_pos"]
            chunk_j = entry["orig_chunk"]
            ner_j = entry["orig_ner"]
            
            # If this is the first subword, or if it differs from the last token we added,
            # treat it as a new original token occurrence.
            if (not dedup_orig_tokens 
                or token_j != dedup_orig_tokens[-1]["token"]
                or pos_j != dedup_orig_tokens[-1]["pos"]
                or chunk_j != dedup_orig_tokens[-1]["chunk"]
                or ner_j != dedup_orig_tokens[-1]["ner"]):
                
                dedup_orig_tokens.append({
                    "token": token_j,
                    "pos": pos_j,
                    "chunk": chunk_j,
                    "ner": ner_j
                })
            else:
                # It's the same token text & same labels as the last one in dedup_orig_tokens
                # meaning it's just another subword of the same original token.
                # So do nothing, skip it.
                pass
        
        # --------------------------------------------------
        # 3) Create a "sentence" by joining deduplicated tokens
        # --------------------------------------------------
        sentence = " ".join(d["token"] for d in dedup_orig_tokens)
        
        # For convenience, also store the deduplicated tokens & labels in parallel lists
        orig_tokens = [d["token"] for d in dedup_orig_tokens]
        orig_pos = [d["pos"] for d in dedup_orig_tokens]
        orig_chunk = [d["chunk"] for d in dedup_orig_tokens]
        orig_ner = [d["ner"] for d in dedup_orig_tokens]

        new_data.append({
            "id": f"chunk_{i}",
            
            # De-duplicated original tokens
            "orig_tokens": orig_tokens,
            "orig_pos_tags": orig_pos,
            "orig_chunk_tags": orig_chunk,
            "orig_ner_tags": orig_ner,

            # Text is the "sentence" from deduplicated tokens
            "text": sentence
        })
    
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare CoNLL data by chunking into fixed-size subword sequences')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input CoNLL JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the output JSONL file')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum number of subwords per chunk (default: 256)')
    
    args = parser.parse_args()
    
    conll_data = []
    with open(args.input_path, 'r', encoding='utf-8') as file:
        for line in file:
            conll_data.append(json.loads(line))

    chunks = build_fixed_size_chunks(conll_data, args.max_length)
    new_dataset = convert_chunks_to_conll_format(chunks)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved new dataset to {args.output_path}.")
