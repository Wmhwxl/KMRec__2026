# KMRec

**Knowledge Graph-Enhanced Multimodal Recommendation with Sentiment-Aware and Structure-Driven Fusion**

This repository provides the implementation of **KMRec**, a knowledge graph-enhanced multimodal recommendation framework. KMRec improves multimodal recommendation by modeling sentiment-aware user-item interactions and structure-driven item-item relationships within a unified knowledge graph.

## Overview

KMRec is designed for multimodal recommendation scenarios where user preferences are influenced by multiple sources of information, including user-item interactions, item images, textual descriptions, user reviews, and item attributes.

The main idea of KMRec is to construct a knowledge graph that captures two types of structural information:

1. **Sentiment-aware user-item interactions**, where user reviews are used to model fine-grained emotional feedback toward items.
2. **Structure-driven item-item relationships**, where shared item attributes are used to model latent structural connections among items.

Based on the constructed knowledge graph, KMRec further integrates knowledge graph representations with multimodal item features to improve recommendation accuracy, especially in sparse and cold-start scenarios.

## Project Structure

The expected project structure is shown below.

```bash
KMRec/
├── data/
│   ├── baby/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
│   │   ├── image_feat.npy
│   │   ├── text_feat.npy
│   │   ├── item_attributes.json
│   │   ├── reviews.json
│   │   └── kg/
│   │       ├── train_triples.txt
│   │       ├── entity2id.txt
│   │       └── relation2id.txt
│   ├── sports/
│   └── clothing/
│
├── src/
│   ├── main.py
│   ├── models/
│   ├── dataloaders/
│   ├── utils/
│   └── kg/
│
├── scripts/
│   ├── run_kmrec.sh
│   ├── pretrain_triples.sh
│   └── evaluate.sh
│
├── log/
├── requirements.txt
└── README.md
```

If your actual project structure is different, please adjust the paths in the running scripts accordingly.

## Environment Requirements

The recommended environment is:

```bash
python >= 3.8
pytorch >= 2.0
cuda >= 11.0
numpy
scipy
pandas
scikit-learn
tqdm
```

You can install the required packages with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, please install the packages manually according to your local environment.

## Dataset Preparation

KMRec supports the commonly used Amazon multimodal recommendation datasets, including:

* Baby
* Sports
* Clothing

Each dataset should be placed under the `data/` directory. For example, the Baby dataset should be organized as:

```bash
data/baby/
├── train.txt
├── valid.txt
├── test.txt
├── image_feat.npy
├── text_feat.npy
├── item_attributes.json
├── reviews.json
└── kg/
```

The required files include:

* `train.txt`: training user-item interactions.
* `valid.txt`: validation user-item interactions.
* `test.txt`: test user-item interactions.
* `image_feat.npy`: image feature vectors of items.
* `text_feat.npy`: textual feature vectors of items.
* `item_attributes.json`: item attribute information.
* `reviews.json`: user review information.
* `kg/`: constructed knowledge graph files.

The knowledge graph used during training should be constructed only from the training data to avoid information leakage from the validation and test sets.

## Knowledge Graph Construction

KMRec constructs a knowledge graph from user reviews and item attributes.

The knowledge graph contains:

* User entities
* Item entities
* Attribute value entities
* Sentiment-aware user-item relations
* Attribute-driven item-attribute relations

The triples are saved in:

```bash
data/<dataset>/kg/train_triples.txt
```

For example:

```bash
data/baby/kg/train_triples.txt
```

The entity and relation mapping files are saved as:

```bash
data/<dataset>/kg/entity2id.txt
data/<dataset>/kg/relation2id.txt
```

Before training KMRec, please make sure that these files have been correctly generated.

## Knowledge Graph Embedding Pretraining

Before running the recommendation model, you may pretrain knowledge graph embeddings using the constructed triples.

An example script is:

```bash
bash scripts/pretrain_triples.sh
```

Please make sure the dataset name and knowledge graph embedding model in `pretrain_triples.sh` are consistent with the settings used in the main training script.

Example configuration:

```bash
dataset="baby"
kg_model="RotatE"
gpu_id="0"
```

The pretrained knowledge graph embeddings should be saved to the corresponding dataset directory or the configured embedding directory.

## Training KMRec

You can train KMRec by running the provided script:

```bash
bash scripts/run_kmrec.sh
```

An example training script is shown below.

```bash
#!/bin/bash

# Create the log directory if it does not exist
mkdir -p log_4_26_26

# Use the current time as part of the log file name
timestamp=$(date +"%Y%m%d_%H%M%S")

# Set model, dataset, and GPU
model="KMRec"
dataset="baby"
gpu_id="2"

# Set the visible GPU
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Run training and redirect the output to the log file
nohup python main.py \
    -m ${model} \
    -d ${dataset} \
    -g ${gpu_id} \
    --export_emb \
    --emb_tag if_lf \
    > "log_4_26_26/${model}_${dataset}_${timestamp}.log" 2>&1 &

# Print the log file path
echo "The log file is saved to: log_4_26_26/${model}_${dataset}_${timestamp}.log"
```

## Script Explanation

The main variables in the script are:

```bash
model="KMRec"
```

This specifies the recommendation model to run. For KMRec, set it to `KMRec`. If you want to run another baseline model, replace it with the corresponding model name.

```bash
dataset="baby"
```

This specifies the dataset. Available options include:

```bash
baby
sports
clothing
```

```bash
gpu_id="2"
```

This specifies the GPU ID. For example, `gpu_id="2"` means that GPU 2 will be used.

```bash
--export_emb
```

This option exports the learned user and item embeddings after training.

```bash
--emb_tag if_lf
```

This option assigns a tag to the exported embeddings. It is useful for distinguishing embeddings generated by different settings.

## Running on Different Datasets

To run KMRec on the Sports dataset, modify the script as follows:

```bash
model="KMRec"
dataset="sports"
gpu_id="0"
```

Then run:

```bash
bash scripts/run_kmrec.sh
```

To run KMRec on the Clothing dataset, modify the script as follows:

```bash
model="KMRec"
dataset="clothing"
gpu_id="0"
```

Then run:

```bash
bash scripts/run_kmrec.sh
```

## Running Baseline Models

If the repository supports baseline models, you can run a baseline by changing the `model` variable.

For example, to run POWERec on the Baby dataset:

```bash
model="POWERec"
dataset="baby"
gpu_id="2"
```

Then execute:

```bash
bash scripts/run_kmrec.sh
```

Please make sure that the selected model name is consistent with the model name defined in the code.

## Log Files

All training logs are saved in the log directory.

For example:

```bash
log_4_26_26/KMRec_baby_20260426_120000.log
```

The log file records the training process, validation results, test results, and other runtime information.

You can monitor the training process with:

```bash
tail -f log_4_26_26/KMRec_baby_20260426_120000.log
```

Please replace the file name with the actual generated log file.

## Evaluation

During training, the model is evaluated on the validation and test sets. The final test performance is usually reported based on the checkpoint that achieves the best validation performance.

The commonly used evaluation metrics include:

* Recall@K
* NDCG@K
* Precision@K
* MAP@K

where `K` is usually selected from:

```bash
1, 5, 10, 20
```

If a separate evaluation script is provided, you can run:

```bash
bash scripts/evaluate.sh
```

Please check the script and modify the model name, dataset name, checkpoint path, and GPU ID before running evaluation.

## Output Embeddings

If `--export_emb` is enabled, the learned embeddings will be exported after training.

The exported embeddings are usually saved in the configured embedding output directory, with file names containing the model name, dataset name, and embedding tag.

For example:

```bash
embeddings/KMRec_baby_if_lf_user.npy
embeddings/KMRec_baby_if_lf_item.npy
```

Please check the actual save path in the code if the output directory is customized.

## Notes

Please make sure that:

1. The dataset files are placed in the correct directory.
2. The knowledge graph files are generated before training.
3. The knowledge graph is constructed only from the training set.
4. The model name in the running script is consistent with the implementation in the code.
5. The GPU ID is available on your machine.
6. The log directory exists before running the script.

## Citation

If you use this code or find this work helpful, please cite our paper:

```bibtex
@article{kmrec,
  title={Knowledge Graph-Enhanced Multimodal Recommendation with Sentiment-Aware and Structure-Driven Fusion},
  author={},
  journal={},
  year={}
}
```

