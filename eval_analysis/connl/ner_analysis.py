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
        ner_tags = record["orig_ner_tags"]
        
        # 2. Unpack each token, ner, and position
        for pos_id, (token, ner) in enumerate(zip(orig_tokens, ner_tags)):
            all_rows.append({
                "connl_instance_id": connl_instance_id,
                "token": token,
                "ner": ner,
                "position_id": pos_id
            })
                
    df = pd.DataFrame(all_rows)

    ## Labels 
    ner_key = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

    df['ner_name'] = df['ner'].map({v: k for k, v in ner_key.items()})
    # Create ner column
    df['ner_cat'] = df['ner_name'].map(lambda x: 'Person' if x in ['B-PER', 'I-PER'] else
                                               'Organization' if x in ['B-ORG', 'I-ORG'] else
                                               'Location' if x in ['B-LOC', 'I-LOC'] else
                                               'Misc' if x in ['B-MISC', 'I-MISC'] else
                                               'Not NE')


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
    ner_cat_stats = df.groupby('ner_cat')['gen_flag'].mean().reset_index()
    ner_cat_stats = ner_cat_stats.rename(columns={'gen_flag': 'gen_rate'})
    ner_cat_stats['omit_rate'] = 1 - ner_cat_stats['gen_rate']
    ner_cat_stats = ner_cat_stats.sort_values(by='omit_rate', ascending=False)

    # If you want to filter out specific ner categories, adjust the list below as needed
    # ner_cat_stats = ner_cat_stats[~ner_cat_stats['ner_cat'].isin(['INTJ', 'X'])]

    print(ner_cat_stats.to_markdown(tablefmt="grid"))
    ner_cat_stats.to_csv(output_path, index=False)

    # === T - Tests ===
    categories = ner_cat_stats['ner_cat'].unique()

    # Perform pairwise t-tests
    results = []
    for cat1, cat2 in itertools.combinations(categories, 2):
        data1 = df[df['ner_cat'] == cat1]['gen_flag']
        data2 = df[df['ner_cat'] == cat2]['gen_flag'] 
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze NER performance on generated text')
    parser.add_argument('--connl_reference', type=str, required=True,
                        help='Path to the reference Connl JSONL file')
    parser.add_argument('--connl_generated', type=str, required=True,
                        help='Path to the generated JSON file (with reconstructed and reference keys)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the output CSV file')
    
    args = parser.parse_args()
    main(args.connl_reference, args.connl_generated, args.output_path)
