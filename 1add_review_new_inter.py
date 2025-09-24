import pandas as pd
import json
import os
from tqdm import tqdm

def load_meta_data(meta_path):
    """Load details and categories information in meta data"""
    print("Loading meta data...")
    meta_data = {}
    detail_keys = set()
    
    # Count the number of file lines for progress bar
    total_lines = sum(1 for _ in open(meta_path, 'r', encoding='utf-8'))
    
    # First traversal: collect all possible detail keys
    with tqdm(total=total_lines, desc="Collecting detail keys") as pbar:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'details' in item and isinstance(item['details'], dict):
                        detail_keys.update(item['details'].keys())
                except json.JSONDecodeError:
                    continue
                pbar.update(1)
    
    print(f"Found {len(detail_keys)} unique detail keys")
    
    # Second traversal: extract details and categories information
    with tqdm(total=total_lines, desc="Processing items") as pbar:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    asin = item.get('parent_asin', '')
                    if not asin:
                        continue
                    
                    # extract details and categories information
                    details = item.get('details', {})
                    categories = item.get('categories', [])
                    
                    # construct details text
                    details_text = []
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if isinstance(value, list):
                                value = ', '.join(str(v) for v in value)
                            elif value is None:
                                value = ''
                            details_text.append(f"{key}: {value}")
                    
                    # construct categories text
                    categories_text = ' > '.join([cat for cat in categories]) if categories else ''
                    
                    # store both details and categories
                    meta_data[asin] = {
                        'details': ' | '.join(details_text) if details_text else '',
                        'categories': categories_text
                    }
                    
                    # print example
                    if len(meta_data) <= 3:
                        print(f"\nExample for ASIN {asin}:")
                        print("Details:", meta_data[asin]['details'][:200] + '...' if len(meta_data[asin]['details']) > 200 else meta_data[asin]['details'])
                        print("Categories:", meta_data[asin]['categories'])
                    
                    pbar.update(1)
                    pbar.set_postfix(items=len(meta_data))
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    pbar.update(1)
                    continue
    
    print(f"\nLoaded meta data for {len(meta_data)} items")
    return meta_data

def process_dataset(dataset_name):
    """Process a single dataset"""
    base_path = f"./data/{dataset_name}"
    print(f"\nProcessing dataset: {dataset_name}")
    
    # load files
    inter_path = os.path.join(base_path, f"{dataset_name}.inter")
    u_mapping_path = os.path.join(base_path, "u_id_mapping.csv")
    i_mapping_path = os.path.join(base_path, "i_id_mapping.csv")
    json_path = os.path.join(base_path, f"{dataset_name}_5.jsonl")
    meta_path = os.path.join(base_path, f"meta_{dataset_name}.jsonl")
    
    try:
        # load basic data
        inter_df = pd.read_csv(inter_path, delimiter="\t")
        u_id_mapping = pd.read_csv(u_mapping_path, delimiter="\t")
        i_id_mapping = pd.read_csv(i_mapping_path, delimiter="\t")
        
        print(f"Initial data size: {len(inter_df)}")
        
        # load meta data
        meta_data = load_meta_data(meta_path)
        
        # create ID mapping
        user_id_map = u_id_mapping.set_index('userID')['user_id'].to_dict()
        item_id_map = i_id_mapping.set_index('itemID')['asin'].to_dict()
        
        # load review data
        with open(json_path, "r") as file:
            reviews_data = [json.loads(line) for line in file]
        print(f"Loaded reviews: {len(reviews_data)}")
        
        # create review dictionary
        review_dict = {(review['reviewerID'], review['asin']): review['reviewText'] 
                      for review in reviews_data}
        
        # add mapping columns
        inter_df['temp_userID'] = inter_df['userID'].map(user_id_map)
        inter_df['temp_itemID'] = inter_df['itemID'].map(item_id_map)
        
        # add review text
        tqdm.pandas(desc="Adding reviews")
        inter_df['reviewText'] = inter_df.progress_apply(
            lambda row: review_dict.get((row['temp_userID'], row['temp_itemID']), ''),
            axis=1
        )
        
        # add details and categories information
        print("\nAdding details and categories information...")
        inter_df['details'] = inter_df['temp_itemID'].apply(
            lambda x: meta_data.get(x, {}).get('details', '')
        )
        inter_df['categories'] = inter_df['temp_itemID'].apply(
            lambda x: meta_data.get(x, {}).get('categories', '')
        )
        
        # create final DataFrame, add categories column
        new_inter = inter_df[['userID', 'itemID', 'rating', 'timestamp', 'x_label', 
                             'reviewText', 'details', 'categories']]
        
        # save results
        output_path = os.path.join(base_path, f"new_{dataset_name}.inter")
        new_inter.to_csv(
            output_path,
            sep='\t',
            index=False,
            quoting=3,
            escapechar="\\",
            na_rep=''
        )
        
        # verify results
        print("\nVerification results:")
        print(f"Original rows: {len(inter_df)}")
        print("\nSample of details and categories:")
        print("\nDetails sample:")
        print(new_inter['details'].head())
        print("\nCategories sample:")
        print(new_inter['categories'].head())
        
        # print statistics
        non_empty_details = new_inter['details'].str.strip().str.len() > 0
        non_empty_categories = new_inter['categories'].str.strip().str.len() > 0
        print(f"\nRows with details: {non_empty_details.sum()} ({non_empty_details.mean():.2%})")
        print(f"Rows with categories: {non_empty_categories.sum()} ({non_empty_categories.mean():.2%})")
        
        return True
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        return False

def main():
    # define the list of datasets to process
    datasets = ['elec']  # add other dataset names
    
    results = {}
    with tqdm(datasets, desc="Processing datasets", unit="dataset") as pbar:
        for dataset in pbar:
            pbar.set_description(f"Processing {dataset}")
            print(f"\n{'='*50}")
            print(f"Processing {dataset} dataset")
            print(f"{'='*50}")
            
            success = process_dataset(dataset)
            results[dataset] = success
            
            # update progress bar description
            status = "✓" if success else "✗"
            pbar.set_postfix(status=status)
    
    # print summary report
    print("\n\nProcessing Summary:")
    print("="*30)
    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{dataset}: {status}")

if __name__ == "__main__":
    main()
