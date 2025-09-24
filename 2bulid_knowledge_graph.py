import json
import torch
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline

class KnowledgeGraphBuilder:
    def __init__(self):
        # loading sentiment analysis model
        model_dir = "./model_cache/sentiment_model"
        self.sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model=model_dir,
            tokenizer=model_dir,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # sentiment score mapping
        self.sentiment_mapping = {
            1: "very_negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "very_positive"
        }
        
        self.batch_size = 32

    def extract_item_features(self, details: str, categories: str) -> Dict[str, List[str]]:
        """extract item features from details and categories"""
        features = {
            "attributes": [],
            "categories": []
        }
        
        # process details
        if details:
            detail_pairs = details.split(' | ')
            for pair in detail_pairs:
                try:
                    key, value = pair.split(': ', 1)
                    # use spaCy to process value to extract keywords
                    doc = self.nlp(value)
                    # extract noun phrases
                    for chunk in doc.noun_chunks:
                        if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
                            features["attributes"].append({
                                "type": key.strip(),
                                "value": chunk.text.strip()
                            })
                except ValueError:
                    continue
        
        # process categories
        if categories:
            cat_hierarchy = categories.split(' > ')
            features["categories"] = [{"level": i, "category": cat.strip()} 
                                   for i, cat in enumerate(cat_hierarchy)]
        
        return features

    def analyze_sentiment(self, review: str) -> Dict:
        """analyze review sentiment"""
        try:
            if not review or not isinstance(review, str):
                return {"sentiment": "neutral", "score": 3}
            
            # truncate long review
            truncated_review = ' '.join(review.split()[:100])
            sentiment_result = self.sentiment_analyzer(
                truncated_review,
                truncation=True,
                max_length=512
            )[0]
            
            score = int(sentiment_result['label'][0])
            return {
                "sentiment": self.sentiment_mapping[score],
                "score": score
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "score": 3}



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """clean dataset"""
    print("Cleaning dataset...")
    
    # 1. ensure the type of numerical columns is correct
    df['userID'] = pd.to_numeric(df['userID'], errors='coerce')
    df['itemID'] = pd.to_numeric(df['itemID'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # 2. ensure the type of string columns is correct
    string_columns = ['reviewText', 'details', 'categories']
    for col in string_columns:
        df[col] = df[col].astype(str)
        # convert 'nan' and 'None' to empty string
        df[col] = df[col].replace({'nan': '', 'None': '', 'NaN': ''})
        # clean special characters
        df[col] = df[col].str.replace(r'\\', r'\\\\', regex=False)
        df[col] = df[col].str.replace(r'\t', ' ', regex=False)
        df[col] = df[col].str.strip()
    
    # 3. remove invalid rows
    df = df.dropna(subset=['userID', 'itemID', 'rating'])
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def analyze_attribute_distribution(df: pd.DataFrame) -> Dict[str, Dict]:
    """analyze attribute distribution"""
    print("\nAnalyzing attribute distribution...")
    attribute_stats = defaultdict(lambda: {"count": 0, "values": defaultdict(int)})
    total_items = 0
    
    with tqdm(total=len(df), desc="Analyzing attributes") as pbar:
        for _, row in df.iterrows():
            if not isinstance(row['details'], str):
                pbar.update(1)
                continue
            
            details = row['details'].strip()
            if not details:
                pbar.update(1)
                continue
            
            total_items += 1
            # analyze each attribute and its value
            for pair in details.split(' | '):
                try:
                    key, value = pair.split(': ', 1)
                    key = key.strip()
                    value = value.strip()
                    if key:
                        attribute_stats[key]["count"] += 1
                        attribute_stats[key]["values"][value] += 1
                except ValueError:
                    continue
            pbar.update(1)
    
    # sort by frequency
    sorted_attrs = sorted(
        attribute_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    # print statistics
    print(f"\nFound {len(attribute_stats)} unique attributes in {total_items} items")
    print("\nTop 20 most common attributes:")
    print("-" * 80)
    print(f"{'Attribute':<30} | {'Count':>8} | {'Coverage':>8} | {'Unique Values':>12}")
    print("-" * 80)
    
    for attr, stats in sorted_attrs[:20]:
        coverage = (stats["count"] / total_items) * 100
        unique_values = len(stats["values"])
        print(f"{attr[:30]:<30} | {stats['count']:>8d} | {coverage:>7.2f}% | {unique_values:>12d}")
        
        # show the top 5 most common values of the attribute
        if stats["values"]:
            print("  Top 5 most common values:")
            top_values = sorted(stats["values"].items(), key=lambda x: x[1], reverse=True)[:5]
            for value, count in top_values:
                value_coverage = (count / stats["count"]) * 100
                print(f"    - {value[:50]}: {count} ({value_coverage:.1f}%)")
        print()
    
    return dict(sorted_attrs)

def build_dataset_knowledge_graphs(dataset_path: str, output_path: str):
    """build knowledge graphs for the entire dataset"""
    builder = KnowledgeGraphBuilder()
    
    print("Reading dataset...")
    try:
        # read dataset
        df = pd.read_csv(
            dataset_path,
            delimiter='\t',
            quoting=3,
            doublequote=False,
            escapechar='\\',
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python'
        )
        
        # clean dataset
        df = clean_dataset(df)
        
        # add attribute distribution analysis
        print("\nAnalyzing attribute distribution before building knowledge graphs...")
        attribute_stats = analyze_attribute_distribution(df)
        
        # show dataset information
        print("\nDataset info:")
        print(df.info())
        print("\nSample details:")
        print(df['details'].head())
        print("\nSample categories:")
        print(df['categories'].head())
        
        knowledge_graphs = []
        failed_rows = 0
        
        # build knowledge graphs for each interaction
        print("\nBuilding knowledge graphs...")
        with tqdm(total=len(df), desc="Processing items") as pbar:
            for idx, row in df.iterrows():
                try:
                    if not isinstance(row['details'], str) or not isinstance(row['categories'], str):
                        print(f"Invalid types at row {idx}: details({type(row['details'])}), categories({type(row['categories'])})")
                        failed_rows += 1
                        continue
                        
                    graph = {
                        "item_id": int(row['itemID']),
                        "user_id": int(row['userID']),
                        "rating": float(row['rating']),
                        "features": {
                            "attributes": [],
                            "categories": []
                        },
                        "sentiment": builder.analyze_sentiment(row['reviewText'])
                    }
                    
                    # process details
                    if row['details'].strip():
                        for pair in row['details'].split(' | '):
                            try:
                                key, value = pair.split(': ', 1)
                                graph['features']['attributes'].append({
                                    "type": key.strip(),
                                    "value": value.strip()
                                })
                            except ValueError:
                                continue
                    
                    # process categories
                    if row['categories'].strip():
                        categories = row['categories'].split(' > ')
                        graph['features']['categories'] = [
                            {"level": i, "category": cat.strip()}
                            for i, cat in enumerate(categories)
                            if cat.strip()
                        ]
                    
                    knowledge_graphs.append(graph)
                    
                except Exception as e:
                    print(f"Error at row {idx}: {str(e)[:200]}")
                    failed_rows += 1
                    continue
                
                pbar.update(1)
        
        # save results
        print("\nSaving knowledge graphs...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for graph in knowledge_graphs:
                json_str = json.dumps(graph, ensure_ascii=False)
                f.write(json_str + '\n')
        
        # print statistics
        print(f"\nProcessing Summary:")
        print(f"Total rows: {len(df)}")
        print(f"Successfully processed: {len(knowledge_graphs)}")
        print(f"Failed rows: {failed_rows}")
        print(f"Success rate: {len(knowledge_graphs)/len(df):.2%}")
        
        # verify output
        print("\nVerifying output...")
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print("\nSample output:")
            print(json.dumps(json.loads(first_line), indent=2))
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    args = parser.parse_args()
    
    dataset_path = f"./data/{args.dataset}/new_{args.dataset}.inter"
    output_path = f"./data/{args.dataset}/knowledge_graphs_{args.dataset}_421.json"

    
    build_dataset_knowledge_graphs(dataset_path, output_path)
