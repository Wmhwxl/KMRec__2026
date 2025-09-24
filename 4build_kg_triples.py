import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

class TripleBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_dir = f"./data/{dataset_name}/kg"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # file paths
        self.entity2id_file = os.path.join(self.data_dir, 'entity2id.txt')
        self.relation2id_file = os.path.join(self.data_dir, 'relation2id.txt')
        self.triple_file = os.path.join(self.data_dir, 'triples.txt')

    def normalize_entity(self, entity: str) -> str:
        """normalize entity name, remove special characters"""
        # remove special characters, keep letters, numbers, and underscores
        normalized = re.sub(r'[^A-Za-z0-9_]', '_', str(entity))
        # ensure not starting with a number
        if normalized and normalized[0].isdigit():
            normalized = 'n_' + normalized
        # remove consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        # remove leading and trailing underscores
        normalized = normalized.strip('_')
        return normalized if normalized else 'unknown'

    def build_triples(self, input_file: str):
        """build triples from structured knowledge graphs"""
        print(f"Building triples from {input_file}")
        
        # load dataset split information
        inter_file = os.path.join("./data", self.dataset_name, f"new_{self.dataset_name}.inter")
        split_dict = {}  # store user-item split information
        
        print("Loading dataset split information...")
        with open(inter_file, 'r', encoding='utf-8') as f:
            header = next(f).strip().split('\t')  # read header
            # find the index of the required columns
            user_idx = header.index('userID')
            item_idx = header.index('itemID')
            label_idx = header.index('x_label')

            for line in f:
                values = line.strip().split('\t')
                user_id = values[user_idx]
                item_id = values[item_idx]
                x_label = values[label_idx]
                split_dict[f"user_{user_id}-item_{item_id}"] = int(x_label)
        
        # read structured knowledge graph
        with open(input_file, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        # collect all entities and relations, and record the mapping from original values to normalized values
        entities = set()
        relations = {'interacts_with', 'belongs_to'}
        triples = []
        entity_mapping = {}  # record the mapping from original values to normalized values
        
        # process all interactions
        for user_data in tqdm(kg_data.values(), desc="Processing users"):
            raw_uid = user_data['user_id']
            user_id = raw_uid if raw_uid.startswith('user_') else f"user_{raw_uid}"
            normalized_uid = user_id  # user ID remains the same
            entities.add(normalized_uid)
            entity_mapping[raw_uid] = normalized_uid
            
            for interaction in user_data['interactions']:
                # process item ID
                raw_iid = interaction['item_id']
                item_id = raw_iid if raw_iid.startswith('item_') else f"item_{raw_iid}"
                normalized_iid = item_id  # item ID remains the same
                entities.add(normalized_iid)
                entity_mapping[raw_iid] = normalized_iid
                
                # merge interaction and sentiment into a single triple
                if 'sentiment' in interaction and interaction['sentiment']:
                    # use feels_{sentiment} when there is sentiment information
                    raw_sentiment = interaction['sentiment']
                    normalized_sentiment = self.normalize_entity(raw_sentiment)
                    relations.add(f'feels_{normalized_sentiment}')
                    relations.add(f'receives_{normalized_sentiment}')
                    triples.append((normalized_uid, f'feels_{normalized_sentiment}', normalized_iid))
                    triples.append((normalized_iid, f'receives_{normalized_sentiment}', normalized_uid))
                    # triples.append((normalized_uid, f'feels_{normalized_sentiment}', normalized_iid))
                    # triples.append((normalized_iid, f'receives_{normalized_sentiment}', normalized_uid))
                else:
                    # use interacts_with when there is no sentiment information
                    relations.add('interacts_with')
                    triples.append((normalized_uid, 'interacts_with', normalized_iid))
                    triples.append((normalized_iid, 'interacts_with', normalized_uid))
                
                # process attributes
                for attr_key, attr_value in interaction['attributes'].items():
                    # normalize attribute name as a relation
                    normalized_key = self.normalize_entity(attr_key)
                    relations.add(normalized_key)
                    
                    # normalize attribute value as an entity
                    normalized_value = self.normalize_entity(attr_value)
                    entities.add(normalized_value)
                    entity_mapping[attr_value] = normalized_value
                    
                    # triples.append((normalized_iid, normalized_key, normalized_value))
                
                # process categories
                for category in interaction['categories']:
                    normalized_category = self.normalize_entity(category)
                    entities.add(normalized_category)
                    entity_mapping[category] = normalized_category
                    triples.append((normalized_iid, 'belongs_to', normalized_category))
        
        # build ID mapping
        entity2id = {e: i for i, e in enumerate(sorted(entities))}
        relation2id = {r: i for i, r in enumerate(sorted(relations))}
        
        # before splitting the dataset, save the complete triples
        # save complete triples.txt
        with open(self.triple_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(triples)}\n")
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

        # then split the dataset
        train_triples = []
        valid_triples = []
        test_triples = []
        
        for h, r, t in triples:
            if h.startswith('user_') and t.startswith('item_') or h.startswith('item_') and t.startswith('user_'):
                # user-item interaction triple
                key = f"{h}-{t}"
                if key in split_dict:
                    split = split_dict[key]
                    if split == 0:
                        # training set retains all triples (including sentiment relations)
                        train_triples.append((h, r, t))
                        train_triples.append((t, r, h))  # add reverse triple
                    else:
                        # validation set and test set only retain interacts_with relations
                        if r == 'interacts_with' or True:
                            if split == 1:
                                valid_triples.append((h, r, t))
                            else:  # split == 2
                                test_triples.append((h, r, t))
            else:
                # other types of triples (attributes, categories, etc.) are put into the training set
                train_triples.append((h, r, t))
        
        # save the split triples
        def save_split(split_triples, filename):
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{len(split_triples)}\n")
                for h, r, t in split_triples:
                    f.write(f"{h}\t{r}\t{t}\n")
        
        save_split(train_triples, 'kg_triples_train_no_attribute.txt')
        save_split(valid_triples, 'kg_triples_valid_no_attribute.txt')
        save_split(test_triples, 'kg_triples_test_no_attribute.txt')
        
        print(f"\nSplit Statistics:")
        print(f"Train triples: {len(train_triples)}")
        print(f"Valid triples: {len(valid_triples)}")
        print(f"Test triples: {len(test_triples)}")
        
        # save entity2id
        with open(self.entity2id_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(entity2id)}\n")
            for entity, eid in sorted(entity2id.items(), key=lambda x: x[1]):
                f.write(f"{entity}\t{eid}\n")
        
        # save relation2id
        with open(self.relation2id_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(relation2id)}\n")
            for relation, rid in sorted(relation2id.items(), key=lambda x: x[1]):
                f.write(f"{relation}\t{rid}\n")
        
        # save entity mapping, for checking the conversion records
        mapping_file = os.path.join(self.data_dir, 'entity_mapping.txt')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            for orig, norm in sorted(entity_mapping.items()):
                if orig != norm:  # only save the mapping that has changed
                    f.write(f"{orig}\t=>\t{norm}\n")
        
        print(f"\nKnowledge Graph Statistics:")
        print(f"Entities: {len(entity2id)}")
        print(f"Relations: {len(relation2id)}")
        print(f"Triples: {len(triples)}")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    args = parser.parse_args()
    
    input_file = f"./data/{args.dataset}/structured_knowledge_graphs_{args.dataset}.json"
    
    builder = TripleBuilder(args.dataset)
    builder.build_triples(input_file)
