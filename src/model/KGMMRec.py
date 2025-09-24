# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

class KGMMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KGMMRec, self).__init__(config, dataset)
        self.sparse = True
        self.config = config
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.align_loss = config['align_loss']
        self.tau = config['tau']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.entity_kg_embedding = nn.Embedding(2*self.embedding_dim, self.embedding_dim)
        self.relation_kg_embedding = nn.Embedding(2*self.embedding_dim, self.embedding_dim)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        if hasattr(dataset, 'entity2id'):
            self.entity2id = dataset.entity2id
        else:
            entity2id_path = os.path.join(
                os.path.abspath(config['data_path']), config['dataset'], 'kg', 'entity2id.txt'
            )
            self.entity2id = {}
            with open(entity2id_path, 'r', encoding='utf-8') as f:
                n_entity = int(f.readline().strip())
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent, idx = parts[0], parts[-1]
                        self.entity2id[ent] = int(idx)
        if hasattr(dataset, 'relation2id'):
            self.relation2id = dataset.relation2id  
        else:
            relation2id_path = os.path.join(
                os.path.abspath(config['data_path']), config['dataset'], 'kg', 'relation2id.txt'
            )
            self.relation2id = {}
            with open(relation2id_path, 'r', encoding='utf-8') as f:
                n_relation = int(f.readline().strip())
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        rel, idx = parts[0], parts[-1]
                        self.relation2id[rel] = int(idx)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        pretrained_kg_dir = os.path.join(
            os.path.abspath(config['pretrained_path']),
            config['dataset'],
            config['kg_model'].lower()
        )

        if config['kg_model'].lower() == 'rotate':
            pretrained_entity_path = os.path.join(pretrained_kg_dir, f"entity_embedding_{self.embedding_dim*2}.pt")
            pretrained_relation_path = os.path.join(pretrained_kg_dir, f"relation_embedding_{self.embedding_dim*2}.pt")
            if os.path.exists(pretrained_entity_path):
                pretrained_entity_embeds = torch.load(pretrained_entity_path, map_location=self.device)
                self.entity_kg_embedding = nn.Embedding.from_pretrained(pretrained_entity_embeds, freeze=False)
                self.entity_kg_mapping = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False).to(self.device)
                user_indices = torch.LongTensor([self.entity2id[f"user_{i}"] for i in range(self.n_users)]).to(self.device)
                item_indices = torch.LongTensor([self.entity2id[f"item_{i}"] for i in range(self.n_items)]).to(self.device)
                user_init = self.entity_kg_mapping(self.entity_kg_embedding(user_indices))
                item_init = self.entity_kg_mapping(self.entity_kg_embedding(item_indices))
                self.user_kg_embedding = nn.Embedding.from_pretrained(user_init, freeze=False)
                self.item_kg_embedding = nn.Embedding.from_pretrained(item_init, freeze=False)
            else:
                self.user_kg_embedding = nn.Embedding(self.n_users, self.embedding_dim)
                self.item_kg_embedding = nn.Embedding(self.n_items, self.embedding_dim)
                nn.init.xavier_uniform_(self.user_kg_embedding.weight)
                nn.init.xavier_uniform_(self.item_kg_embedding.weight)

            if os.path.exists(pretrained_relation_path):
                pretrained_relation_embeds = torch.load(pretrained_relation_path, map_location=self.device)
                self.relation_kg_embedding = nn.Embedding.from_pretrained(pretrained_relation_embeds, freeze=False)
                self.relation_kg_mapping = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False).to(self.device)
                relation_init = self.relation_kg_mapping(self.relation_kg_embedding.weight)
                self.relation_kg_embeds = nn.Embedding.from_pretrained(relation_init, freeze=False)
            else:
                self.relation_kg_embeds = nn.Embedding(dataset.n_relations, self.embedding_dim)
                nn.init.xavier_uniform_(self.relation_kg_embeds.weight)

        else:
            pretrained_entity_path = os.path.join(pretrained_kg_dir, f"entity_embedding_{self.embedding_dim}.pt")
            pretrained_relation_path = os.path.join(pretrained_kg_dir, f"relation_embedding_{self.embedding_dim}.pt")
            if os.path.exists(pretrained_entity_path):
                pretrained_entity_embeds = torch.load(pretrained_entity_path, map_location=self.device)
                user_indices = [self.entity2id[f"user_{i}"] for i in range(self.n_users)]
                item_indices = [self.entity2id[f"item_{i}"] for i in range(self.n_items)]
                user_init = pretrained_entity_embeds[user_indices]
                item_init = pretrained_entity_embeds[item_indices]
                self.user_kg_embedding = nn.Embedding.from_pretrained(user_init, freeze=False)
                self.item_kg_embedding = nn.Embedding.from_pretrained(item_init, freeze=False)
            else:
                self.user_kg_embedding = nn.Embedding(self.n_users, self.embedding_dim)
                self.item_kg_embedding = nn.Embedding(self.n_items, self.embedding_dim)
                nn.init.xavier_uniform_(self.user_kg_embedding.weight)
                nn.init.xavier_uniform_(self.item_kg_embedding.weight)

            if os.path.exists(pretrained_relation_path):
                pretrained_relation_embeds = torch.load(pretrained_relation_path, map_location=self.device)
                self.relation_kg_embeds = nn.Embedding.from_pretrained(pretrained_relation_embeds, freeze=False)
            else:
                self.relation_kg_embeds = nn.Embedding(dataset.n_relations, self.embedding_dim)
                nn.init.xavier_uniform_(self.relation_kg_embeds.weight)

        self.user_kg_embedding = self.user_kg_embedding.to(self.device)
        self.item_kg_embedding = self.item_kg_embedding.to(self.device)

        # pretrained_relation_path = os.path.join(pretrained_kg_dir, f"relation_embedding_{self.embedding_dim*2}.pt")
        # if os.path.exists(pretrained_relation_path):
        #     pretrained_relation_embeds = torch.load(pretrained_relation_path, map_location=self.device)
        #     self.relation_kg_embeds = nn.Embedding.from_pretrained(pretrained_relation_embeds, freeze=False)
        # else:
        #     self.relation_kg_embeds = nn.Embedding(dataset.n_relations, self.embedding_dim)
        #     nn.init.xavier_uniform_(self.relation_kg_embeds.weight)

        self.sentiment_user_embedding, self.sentiment_item_embedding = self.init_sentiment_embeddings()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        self.fusion_adj = self.max_pool_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.user_kg_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),nn.Sigmoid())
        self.item_kg_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),nn.Sigmoid())

        self.view_score = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 3, bias=False)   
        )

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        

    def init_sentiment_embeddings(self):
        sentiment_scores = {
            'feels_very_positive_5': 5,
            'receives_very_positive_5': 10,
            'feels_positive_4': 4,
            'receives_positive_4': 9,
            'feels_neutral_3': 3,
            'receives_neutral_3': 8,
            'feels_negative_2': 2,
            'receives_negative_2': 7,
            'feels_very_negative_1': 1,
            'receives_very_negative_1': 6,
            'interacts_with': 0
        }
        
        items_ones_vector = torch.ones(self.n_items, 1).to(self.device)
        user_ones_vector = torch.ones(self.n_users, 1).to(self.device)
        
        user_sentiment_embedding = torch.zeros(self.n_users, self.embedding_dim).to(self.device)
        item_sentiment_embedding = torch.zeros(self.n_items, self.embedding_dim).to(self.device)
        
        sentiment_matrices_dir = os.path.join(self.config['data_path'], self.config['dataset'], 'sentiment_matrices')
        
        for score in range(11):
            adj_path = os.path.join(sentiment_matrices_dir, f'sentiment_adj_{score}.npz')
            if not os.path.exists(adj_path):
                print(f"Warning: sentiment matrix {score} not found at {adj_path}")
                continue
            adj = sp.load_npz(adj_path)
            
            user_item_adj = adj[:self.n_users, self.n_users:].tocsr()
            item_user_adj = adj[self.n_users:, :self.n_users].tocsr()
            
            user_sentiment = torch.from_numpy(
                user_item_adj.dot(items_ones_vector.cpu().numpy())
            ).float().to(self.device)
            
            item_sentiment = torch.from_numpy(
                item_user_adj.dot(user_ones_vector.cpu().numpy())
            ).float().to(self.device)
            
            relations = [r for r, s in sentiment_scores.items() if s == score]
            if relations:
                relation = relations[0]
                relation_id = self.relation2id[relation]
                relation_embed = self.relation_kg_embeds(
                    torch.tensor([relation_id]).to(self.device)
                )
                
                user_sentiment_embedding += user_sentiment * relation_embed
                item_sentiment_embedding += item_sentiment * relation_embed
        
        user_sentiment_embedding = F.normalize(user_sentiment_embedding, p=2, dim=1)
        item_sentiment_embedding = F.normalize(item_sentiment_embedding, p=2, dim=1)
        
        return nn.Parameter(user_sentiment_embedding), nn.Parameter(item_sentiment_embedding)

    def pre_epoch_processing(self):
        pass

    def max_pool_fusion(self):
        image_adj = self.image_original_adj.coalesce()
        text_adj = self.text_original_adj.coalesce()

        image_indices = image_adj.indices().to(self.device)
        image_values = image_adj.values().to(self.device)
        text_indices = text_adj.indices().to(self.device)
        text_values = text_adj.values().to(self.device)

        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

        combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

        return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    
    def multimodal_fusion(self, text_feats, image_feats):
        """
        u_{fusion} = [e_{text}, e_{text}*e_{vision}, e_{text}-e_{vision}, e_{vision}]
        """
        mult_feat = torch.multiply(text_feats, image_feats)  # element-wise multiplication
        diff_feat = text_feats - image_feats    # difference
        
        fusion_feats = torch.cat([
            text_feats,          
            mult_feat,           
            diff_feat,          
            image_feats        
        ], dim=1)
        
       
        fusion_layer = nn.Linear(self.embedding_dim * 4, self.embedding_dim).to(self.device)
        fused_embeds = fusion_layer(fusion_feats)
        
        return fused_embeds
    
    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        item_scene_embeds = self.item_kg_embedding.weight
        user_scene_embeds = self.user_kg_embedding.weight

        item_embeds = self.item_id_embedding.weight 
        user_embeds = self.user_embedding.weight
        
        user_gate = torch.sigmoid(self.user_kg_gate(user_scene_embeds))
        item_gate = torch.sigmoid(self.item_kg_gate(item_scene_embeds))
        user_embeds = user_gate * user_scene_embeds +  (1 - user_gate) * user_embeds
        item_embeds = item_gate * item_scene_embeds +  (1 - item_gate) * item_embeds
        image_item_embeds = self.multimodal_fusion(item_embeds, self.gate_v(image_feats))
        text_item_embeds = self.multimodal_fusion(item_embeds, self.gate_t(text_feats))
        fused_feats = self.multimodal_fusion(text_feats, image_feats)
        fusion_item_embeds = torch.multiply(item_embeds, self.gate_f(fused_feats))

        #   User-Item (Behavioral) View
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds )
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds )
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds )
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        #   Modality-aware Preference Module
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds) 
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, fusion_embeds, agg_text_embeds, agg_image_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, fusion_embeds, text_embeds, image_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        fusion_embeds_users, fusion_embeds_items = torch.split(fusion_embeds, [self.n_users, self.n_items], dim=0)
        text_embeds_users, text_embeds_items = torch.split(text_embeds, [self.n_users, self.n_items], dim=0)
        image_embeds_users, image_embeds_items = torch.split(image_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], self.tau) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], self.tau)
        align_loss = self.InfoNCE(fusion_embeds_items[pos_items], text_embeds_items[pos_items], self.tau) + self.InfoNCE(
        fusion_embeds_items[pos_items], image_embeds_items[pos_items],self.tau) + self.InfoNCE(image_embeds_items[pos_items], text_embeds_items[pos_items],self.tau) 

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.align_loss * align_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
