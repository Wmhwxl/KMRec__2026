import os
import numpy as np
import scipy.sparse as sp

def generate_sentiment_matrices(data_path, dataset, n_users, n_items, triples_file):
    sentiment_scores = {
            'feels_very_positive_5': 5,
            'receives_very_positive_5': 11,
            'feels_positive_4': 4,
            'receives_positive_4': 10,
            'feels_neutral_3': 3,
            'receives_neutral_3': 9,
            'feels_negative_2': 2,
            'receives_negative_2': 8,
            'feels_very_negative_1': 1,
            'receives_very_negative_1': 7,
            'receives_nothing': 6,
            'feels_nothing': 0
        }
    score2relations = {}
    for k, v in sentiment_scores.items():
        score2relations.setdefault(v, []).append(k)

    sentiment_matrices_dir = os.path.join(data_path, dataset, 'sentiment_matrices')
    os.makedirs(sentiment_matrices_dir, exist_ok=True)

    matrices = {score: sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) for score in range(11)}

    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            head, relation, tail = parts
            if head.startswith('user_'):
                head_idx = int(head.split('_')[1])
            elif head.startswith('item_'):
                head_idx = n_users + int(head.split('_')[1])
            else:
                continue
            if tail.startswith('user_'):
                tail_idx = int(tail.split('_')[1])
            elif tail.startswith('item_'):
                tail_idx = n_users + int(tail.split('_')[1])
            else:
                continue
            for score, rels in score2relations.items():
                if relation in rels:
                    matrices[score][head_idx, tail_idx] = 1.0

    for score in range(11):
        matrices[score] = matrices[score].tocsr()
        out_path = os.path.join(sentiment_matrices_dir, f'sentiment_adj_{score}.npz')
        sp.save_npz(out_path, matrices[score])
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    data_path = "./data"
    dataset = "elec"
    n_users = 192403  
    n_items = 63001   
    triples_file = "./data/elec/kg/kg_triples_train.txt"  

    generate_sentiment_matrices(data_path, dataset, n_users, n_items, triples_file)
