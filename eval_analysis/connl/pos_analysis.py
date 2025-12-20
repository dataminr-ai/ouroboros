import argparse
import itertools
import json
import os
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as smm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def main(connl_reference, connl_generated, output_path):
    # Load generated texts
    with open(connl_generated, 'r') as file:
        data = json.load(file)
    generated_texts = data['reconstructed']
    references_copy = data['reference']

    # Read Connl
    jsonl_filepath = connl_reference
    with open(jsonl_filepath, 'r') as jsonl_file:
        jsonl_data = [json.loads(line) for line in jsonl_file]

    all_rows = []

    for idx, record in enumerate(jsonl_data):
        connl_instance_id = idx
        orig_tokens = record["orig_tokens"]
        pos_tags = record["orig_pos_tags"]
        
        # 2. Unpack each token, ner, and position
        for pos_id, (token, pos) in enumerate(zip(orig_tokens, pos_tags)):
            all_rows.append({
                "connl_instance_id": connl_instance_id,
                "token": token,
                "pos": pos,
                "position_id": pos_id
            })
                
    df = pd.DataFrame(all_rows)

    ## Labels 
    pos_key = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
     'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
     'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
     'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
     'WP': 44, 'WP$': 45, 'WRB': 46}

    df['pos_name'] = df['pos'].map({v: k for k, v in pos_key.items()})

    # Create ner column
    pos_to_cat = {
    # Punctuation & symbols
    '"':   "PUNCT",
    "''":  "PUNCT",
    "``":  "PUNCT",
    "#":   "SYM",
    "$":   "SYM",
    "(":   "PUNCT",
    ")":   "PUNCT",
    ",":   "PUNCT",
    ".":   "PUNCT",
    ":":   "PUNCT",

    # Coordinating conjunction
    "CC":  "CONJ",

    # Numbers
    "CD":  "NUM",
    "LS":  "NUM",

    # Determiners & predeterminers
    "DT":  "DET",
    "PDT": "DET",
    "WDT": "DET",

    # Existential "there"
    "EX":  "ADV",  # "EX" is sometimes categorized separately, but "ADV" is okay

    # Foreign word
    "FW":  "X",

    # Preposition / subordinating conjunction / infinitive marker
    "IN":  "ADP",
    "TO":  "ADP",

    # Adjectives
    "JJ":  "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",

    # Modals
    "MD":  "VERB",

    # Nouns
    "NN":     "NOUN",
    "NNS":    "NOUN",
    "NNP":    "NOUN",  # Proper nouns can be merged into NOUN
    "NNPS":   "NOUN",
    "NN|SYM": "NOUN",  # Special CoNLL variant

    # Possessive marker
    "POS": "PRT",      # "'s" /  "'"

    # Pronouns
    "PRP":  "PRON",
    "PRP$": "PRON",
    "WP":   "PRON",
    "WP$":  "PRON",

    # Adverbs
    "RB":   "ADV",
    "RBR":  "ADV",
    "RBS":  "ADV",
    "WRB":  "ADV",

    # Particles
    "RP":   "PRT",

    # Symbols
    "SYM":  "SYM",

    # Interjection
    "UH":   "INTJ",

    # Verbs
    "VB":   "VERB",
    "VBD":  "VERB",
    "VBG":  "VERB",
    "VBN":  "VERB",
    "VBP":  "VERB",
    "VBZ":  "VERB"
    }

    df['pos_cat'] = df['pos_name'].map(pos_to_cat)

    # === Reference Tuples ===
    # Calculate the 1-based cumulative count for each token within each sentence
    counts = df.groupby(['connl_instance_id', 'token']).cumcount() + 1 
    df['token_count_tuple'] = list(zip(df['token'], counts))

    # === Generation Tuples ===
    gen_rows = []
    for connl_instance_id, gen_text in enumerate(generated_texts):
        tokens = nltk.word_tokenize(gen_text)
        for pos_id, token in enumerate(tokens):
            gen_rows.append({
                "connl_instance_id": connl_instance_id,
                "token": token,
                "position_id": pos_id
            })

    gen_df = pd.DataFrame(gen_rows)

    # Create tuples
    gen_counts = gen_df.groupby(['connl_instance_id', 'token']).cumcount() + 1 
    gen_df['token_count_tuple'] = list(zip(gen_df['token'], gen_counts))

    # === Generation Match Flag ===
    match_columns = ['connl_instance_id', 'token_count_tuple']
    ref_index = pd.MultiIndex.from_frame(df[match_columns])
    gen_lookup_index = pd.MultiIndex.from_frame(gen_df[match_columns])

    # Check which rows in df exist in the lookup_set and convert boolean to 0 or 1
    df['gen_flag'] = ref_index.isin(gen_lookup_index).astype(int)

    # === Omit Rate ===
    pos_cat_stats = df.groupby('pos_cat')['gen_flag'].mean().reset_index()
    pos_cat_stats = pos_cat_stats.rename(columns={'gen_flag': 'gen_rate'})
    pos_cat_stats['omit_rate'] = 1 - pos_cat_stats['gen_rate']
    pos_cat_stats = pos_cat_stats.sort_values(by='omit_rate', ascending=False)

    pos_cat_stats = pos_cat_stats[~pos_cat_stats['pos_cat'].isin(['INTJ', 'X'])]

    print(pos_cat_stats.to_markdown(tablefmt="grid"))
    pos_cat_stats.to_csv(output_path, index=False)

    # === T - Tests ===
    categories = pos_cat_stats['pos_cat'].unique()

    # Perform pairwise t-tests
    results = []
    for cat1, cat2 in itertools.combinations(categories, 2):
        data1 = df[df['pos_cat'] == cat1]['gen_flag']
        data2 = df[df['pos_cat'] == cat2]['gen_flag'] 
        # Perform t-test (assuming equal variance)
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
        results.append((cat1, cat2, t_stat, p_value))

    results_df = pd.DataFrame(results, columns=['Category 1', 'Category 2', 't-statistic', 'p-value'])

    # Apply Bonferroni correction for multiple comparisons
    adjusted_p_values = smm.multipletests(results_df['p-value'], method='bonferroni')[1]
    results_df['adjusted p-value'] = adjusted_p_values

    # Find statistically significant pairs
    sig = results_df[results_df['adjusted p-value'] < 0.05]
    sig['Pairs'] = sig['Category 1'] + ' - ' + sig['Category 2']
    sig.drop(columns=['Category 1', 'Category 2'], inplace=True)

    print(sig.to_markdown(tablefmt="grid"))

    # === Determiners Example ===
    det_omitted = df[(df['pos_cat'] == 'DET') & (df['gen_flag'] == 0)]

    # For each unique connl_instance_id, print the corresponding reference and generated text
    for connl_id in det_omitted['connl_instance_id'].unique():
        det_row = det_omitted[det_omitted['connl_instance_id'] == connl_id].iloc[0]
        print(f"\nconnl_instance_id: {connl_id}")
        print("Omitted Token:", det_row['token'])
        print("Reference:")
        print(references_copy[connl_id])
        print("Generated:")
        print(generated_texts[connl_id])
        break

    # gen_df filtered
    gen_df_filtered = gen_df[(gen_df['connl_instance_id'] == 1) & (gen_df['token'].str.lower() == 'the')]
    print(gen_df_filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze POS performance on generated text')
    parser.add_argument('--connl_reference', type=str, required=True,
                        help='Path to the reference Connl JSONL file')
    parser.add_argument('--connl_generated', type=str, required=True,
                        help='Path to the generated JSON file (with reconstructed and reference keys)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the output CSV file')
    
    args = parser.parse_args()
    main(args.connl_reference, args.connl_generated, args.output_path)
