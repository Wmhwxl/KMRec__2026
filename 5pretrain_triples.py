import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import argparse
from kg_models.OpenKE.openke.module.model import ComplEx, TransE, DistMult, Analogy, TransH, TransR, TransD, RotatE, HolE, RESCAL
from tqdm import tqdm


def load_and_split_kg_data(entity_path, relation_path, triple_path, test_ratio=0.1, batch_size=1024, shuffle=True):
    """dataset loading and splitting
    Args:
        entity_path:  entity_id path
        relation_path: relation_id path
        triple_path: triple path
        test_ratio: validation set ratio
        batch_size: batch size
        shuffle: whether to shuffle training data
    Returns:
        train_loader, test_loader, n_entity, n_relation
    """
    # Load entities
    entity2id = {}
    with open(entity_path, 'r', encoding='utf-8') as f:
        n_entity = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ent, idx = parts[0], parts[-1]
                entity2id[ent] = int(idx)

    # Load relations
    relation2id = {}
    with open(relation_path, 'r', encoding='utf-8') as f:
        n_relation = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rel, idx = parts[0], parts[-1]
                relation2id[rel] = int(idx)

    # Load triples
    triples = []
    with open(triple_path, 'r', encoding='utf-8') as f:
        n_triple = int(f.readline().strip())
        for line in f:
            try:
                h, r, t = line.strip().split('\t')
                if h in entity2id and r in relation2id and t in entity2id:
                    triples.append((
                        entity2id[h],
                        relation2id[r],
                        entity2id[t]
                    ))
            except Exception as e:
                continue

    # Split into train and test sets
    random.shuffle(triples)
    split_idx = int(len(triples) * (1 - test_ratio))
    train_triples = triples[:split_idx]
    test_triples = triples[split_idx:]

    train_tensor = torch.LongTensor(train_triples)
    test_tensor = torch.LongTensor(test_triples)

    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(entity2id)} entities, {len(relation2id)} relations, {len(triples)} triples")
    print(f"Train triples: {len(train_triples)}, Test triples: {len(test_triples)}")
    return train_loader, test_loader, n_entity, n_relation


def negative_sample(triples, n_entity):
    """Generate negative samples by corrupting head or tail entities
    Args:
        triples: Tensor of shape (batch_size, 3)
        n_entity: Total number of entities
    Returns:
        neg_triples: Corrupted triples with same shape as input
    """
    device = triples.device  # Get device of input triples
    neg_triples = triples.clone()
    
    # Generate mask and random entities on the same device
    mask = torch.rand(len(triples), device=device) < 0.5
    rand_entities = torch.randint(0, n_entity, (len(triples),), device=device)
    
    # Corrupt either head or tail entity
    neg_triples[mask, 0] = rand_entities[mask]  # Corrupt head entities
    neg_triples[~mask, 2] = rand_entities[~mask]  # Corrupt tail entities
    
    return neg_triples


def compute_loss(model, pos_triples, n_entity, margin=1.0):
    neg_triples = negative_sample(pos_triples, n_entity).to(pos_triples.device)
    if isinstance(model, ComplEx):
        # ComplEx
        h_pos_re = model.ent_re_embeddings(pos_triples[:, 0])
        r_pos_re = model.rel_re_embeddings(pos_triples[:, 1])
        t_pos_re = model.ent_im_embeddings(pos_triples[:, 2])       
        h_neg_re = model.ent_re_embeddings(neg_triples[:, 0])
        r_neg_re = model.rel_re_embeddings(neg_triples[:, 1])     
        t_neg_re = model.ent_im_embeddings(neg_triples[:, 2])
        h_pos_im = model.ent_im_embeddings(pos_triples[:, 0])
        r_pos_im = model.rel_im_embeddings(pos_triples[:, 1])     
        t_pos_im = model.ent_re_embeddings(pos_triples[:, 2])
        h_neg_im = model.ent_im_embeddings(neg_triples[:, 0])   
        r_neg_im = model.rel_im_embeddings(neg_triples[:, 1])
        t_neg_im = model.ent_re_embeddings(neg_triples[:, 2])   
        # _calc
        pos_score = model._calc(h_pos_re, h_pos_im, t_pos_re, t_pos_im, r_pos_re, r_pos_im)
        neg_score = model._calc(h_neg_re, h_neg_im, t_neg_re, t_neg_im, r_neg_re, r_neg_im)
        loss = F.relu(margin - pos_score + neg_score).mean()

    elif isinstance(model, (TransE, DistMult, TransH, TransR, TransD, RotatE, HolE)):
        h_pos = model.ent_embeddings(pos_triples[:, 0])
        r_pos = model.rel_embeddings(pos_triples[:, 1])
        t_pos = model.ent_embeddings(pos_triples[:, 2])
        h_neg = model.ent_embeddings(neg_triples[:, 0])
        r_neg = model.rel_embeddings(neg_triples[:, 1])
        t_neg = model.ent_embeddings(neg_triples[:, 2])
        # _calc
        pos_score = model._calc(h_pos, t_pos, r_pos, "normal")
        neg_score = model._calc(h_neg, t_neg, r_neg, "normal")
        loss = F.relu(margin - pos_score + neg_score).mean()
    elif isinstance(model, Analogy):        
        h_pos_re = model.ent_re_embeddings(pos_triples[:, 0])
        r_pos_re = model.rel_re_embeddings(pos_triples[:, 1])
        t_pos_re = model.ent_re_embeddings(pos_triples[:, 2])
        h_neg_re = model.ent_re_embeddings(neg_triples[:, 0])
        r_neg_re = model.rel_re_embeddings(neg_triples[:, 1])
        t_neg_re = model.ent_re_embeddings(neg_triples[:, 2])
        h_pos_ent = model.ent_embeddings(pos_triples[:, 0])
        r_pos_ent = model.rel_embeddings(pos_triples[:, 1]) 
        t_pos_ent = model.ent_embeddings(pos_triples[:, 2])
        h_neg_ent = model.ent_embeddings(neg_triples[:, 0])
        r_neg_ent = model.rel_embeddings(neg_triples[:, 1])
        t_neg_ent = model.ent_embeddings(neg_triples[:, 2])
        h_pos_im = model.ent_im_embeddings(pos_triples[:, 0])
        r_pos_im = model.rel_im_embeddings(pos_triples[:, 1])
        t_pos_im = model.ent_im_embeddings(pos_triples[:, 2])
        h_neg_im = model.ent_im_embeddings(neg_triples[:, 0])
        r_neg_im = model.rel_im_embeddings(neg_triples[:, 1])
        t_neg_im = model.ent_im_embeddings(neg_triples[:, 2])
        # _calc
        pos_score = model._calc(h_pos_re, h_pos_im, h_pos_ent, t_pos_re, t_pos_im, t_pos_ent, r_pos_re, r_pos_im, r_pos_ent)
        neg_score = model._calc(h_neg_re, h_neg_im, h_neg_ent, t_neg_re, t_neg_im, t_neg_ent ,r_neg_re, r_neg_im, r_neg_ent)
        loss = F.relu(margin - pos_score + neg_score).mean()

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")  
    return loss



def save_model_embeddings(model, save_dir):
    """save model embeddings to specified directory"""
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    if isinstance(model, ComplEx):
        torch.save(model.ent_re_embeddings.weight.data.cpu(), 
                  os.path.join(save_dir, f"entity_re_embedding_{model.ent_re_embeddings.weight.shape[1]}.pt"))
        torch.save(model.ent_im_embeddings.weight.data.cpu(), 
                  os.path.join(save_dir, f"entity_im_embedding_{model.ent_im_embeddings.weight.shape[1]}.pt"))
        torch.save(model.rel_re_embeddings.weight.data.cpu(), 
                  os.path.join(save_dir, f"relation_re_embedding_{model.rel_re_embeddings.weight.shape[1]}.pt"))
        torch.save(model.rel_im_embeddings.weight.data.cpu(), 
                  os.path.join(save_dir, f"relation_im_embedding_{model.rel_im_embeddings.weight.shape[1]}.pt"))
    else:
        torch.save(model.ent_embeddings.weight.data.cpu(),
                  os.path.join(save_dir, f"entity_embedding_{model.ent_embeddings.weight.shape[1]}.pt"))
        torch.save(model.rel_embeddings.weight.data.cpu(),
                  os.path.join(save_dir, f"relation_embedding_{model.ent_embeddings.weight.shape[1]}.pt"))

def train_kg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, n_entity, n_relation = load_and_split_kg_data(
        args.entity2id,
        args.relation2id,
        args.train_triples,
        test_ratio=0.3,  
        batch_size=args.batch_size,
        shuffle=True
    )

    # initialize model
    if args.model == "ComplEx":
        model = ComplEx(n_entity, n_relation, args.dim).to(device)
    elif args.model == "TransE":
        model = TransE(n_entity, n_relation, args.dim).to(device)
    elif args.model == "Analogy":
        model = Analogy(n_entity, n_relation, args.dim).to(device)
    elif args.model == "DistMult":
        model = DistMult(n_entity, n_relation, args.dim).to(device)
    elif args.model == "TransH":
        model = TransH(n_entity, n_relation, args.dim).to(device)
    elif args.model == "TransR":
        model = TransR(n_entity, n_relation, args.dim).to(device)
    elif args.model == "TransD":
        model = TransD(n_entity, n_relation, args.dim).to(device)
    elif args.model == "RotatE":
        model = RotatE(n_entity, n_relation, args.dim).to(device)
    elif args.model == "HolE":
        model = HolE(n_entity, n_relation, args.dim).to(device)
    elif args.model == "RESCAL":
        model = RESCAL(n_entity, n_relation, args.dim).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')
    best_epoch = -1  # optional: Track best epoch

    print("Start training KG embeddings...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            loss = compute_loss(model, batch, n_entity, args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                loss = compute_loss(model, batch, n_entity, args.margin)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        print(f"[Epoch {epoch+1}/{args.epochs}] Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  
            save_dir = os.path.join(args.output_dir, args.model.lower())
            save_model_embeddings(model, save_dir)
            print(f"Saved best model (val_loss={best_val_loss:.4f}) at epoch {best_epoch}")

        if (epoch + 1) % 10 == 0:
            save_dir = os.path.join(args.output_dir, args.model.lower(), f"epoch_{epoch+1}")
            save_model_embeddings(model, save_dir)
            print(f"Saved embedding at epoch {epoch+1}")

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--model', type=str, default='TransE',
    choices=['ComplEx', 'TransE', 'DistMult', 'RESCAL', 'Analogy', 'TransH', 'TransR', 'TransD', 'SimplE', 'RotatE', 'HolE'])
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--margin', type=float, default=1.0)
    
    args = parser.parse_args()

    kg_dir = f"./data/{args.dataset}/kg"
    args.entity2id = os.path.join(kg_dir, "entity2id.txt")
    args.relation2id = os.path.join(kg_dir, "relation2id.txt") 
    args.train_triples = os.path.join(kg_dir, "kg_triples_train_no_attribute.txt")
    args.output_dir = f"./pretrained_no_attribute/{args.dataset}"
    
    train_kg(args)
