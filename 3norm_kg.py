import json
from collections import defaultdict
from tqdm import tqdm
import os

def normalize_attr_key(k: str) -> str:
    """normalize attribute key"""
    return k.lower().strip().replace(" ", "_").replace("-", "_")

def normalize_attr_value(v: str) -> str:
    """normalize attribute value"""
    return v.strip().replace(" ", "_").replace(";", "").replace(",", "").replace("/", "_")

def transform_knowledge_graphs(input_file: str, output_file: str):
    """transform knowledge graphs, keep readable JSON format"""
    print(f"Reading from {input_file}...")
    
    user_interactions = {}
    total_lines = 0
    processed_lines = 0
    
    try:
        # read and process data
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing lines"):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    uid = f"user_{data['user_id']}"
                    
                    if uid not in user_interactions:
                        user_interactions[uid] = {
                            "user_id": uid,
                            "interactions": []
                        }
                    
                    interaction = {
                        "item_id": f"item_{data['item_id']}",
                        "rating": float(data['rating']),
                        "sentiment": f"{data['sentiment']['sentiment']}_{data['sentiment']['score']}",
                        "attributes": {},
                        "categories": []
                    }
                    
                    # process attributes
                    for attr in data['features']['attributes']:
                        if 'type' in attr and 'value' in attr:
                            key = normalize_attr_key(attr['type'])
                            value = normalize_attr_value(attr['value'])
                            if key and value:  # 确保键值对不为空
                                interaction['attributes'][key] = value
                    
                    # process categories
                    skip_categories = {"Clothing, Shoes & Jewelry", "Sport Specific Clothing"}
                    categories = [
                        normalize_attr_value(c["category"])
                        for c in data['features']['categories']
                        if c["category"] not in skip_categories and c["category"].strip()
                    ]
                    interaction['categories'] = categories
                    
                    user_interactions[uid]["interactions"].append(interaction)
                    processed_lines += 1
                    
                except Exception as e:
                    print(f"Error processing line: {str(e)[:200]}")
                    continue
        
        # save as formatted JSON
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(user_interactions, f_out, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing Summary:")
        print(f"Total lines read: {total_lines}")
        print(f"Successfully processed: {processed_lines}")
        print(f"Total users: {len(user_interactions)}")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    args = parser.parse_args()
    
    input_file = f"./data/{args.dataset}/knowledge_graphs_{args.dataset}_421.json"
    output_file = f"./data/{args.dataset}/structured_knowledge_graphs_{args.dataset}.json"
    
    if transform_knowledge_graphs(input_file, output_file):
        print("Transformation completed successfully!")
    else:
        print("Transformation failed!")
