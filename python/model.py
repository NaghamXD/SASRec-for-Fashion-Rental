import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
            super(SASRec, self).__init__()

            self.user_num = user_num
            self.item_num = item_num
            self.dev = args.device
            self.norm_first = args.norm_first

            # TODO: loss += args.l2_emb for regularizing embedding vectors during training
            # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
            self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)

            # --- DYNAMIC FEATURE LOADING (Items vs Groups) ---
            # Determine paths based on dataset name
            if args.dataset and 'group' in args.dataset:
                img_path = 'data/pretrained_group_emb_1280.npy'
                tag_path = 'data/pretrained_group_tag_emb.npy'
                print(f"--> Detected GROUP dataset ({args.dataset}). Loading Group embeddings.")
            else:
                img_path = 'data/pretrained_item_emb_1280.npy'
                tag_path = 'data/pretrained_tag_emb.npy'
                print(f"--> Detected ITEM dataset ({args.dataset}). Loading Item embeddings.")

            # --- NEW: Learnable weights for feature fusion ---
            # We initialize them at 0.1 so the model starts by relying 
            # mostly on IDs and slowly learns how much to trust content.
            self.alpha_visual = torch.nn.Parameter(torch.tensor(0.1).to(self.dev))
            self.alpha_tags = torch.nn.Parameter(torch.tensor(0.1).to(self.dev))

            # =================================================
            # FEATURE 1: IMAGES (EfficientNet)
            # =================================================
            self.visual_dim = 1280
            # Shrink Layer: 1280 -> 50
            self.dim_reduction_layer = torch.nn.Linear(self.visual_dim, args.hidden_units)
            
            # CHANGE: Add LayerNorm for Visual Features
            self.visual_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
    
            # Fixed Lookup Table
            self.visual_features = torch.nn.Embedding(self.item_num+1, self.visual_dim, padding_idx=0)
            
            try:
                weights = np.load(img_path)
                # Safety Check: Does file size match model size?
                if weights.shape[0] == self.item_num + 1:
                    self.visual_features.weight.data.copy_(torch.from_numpy(weights))
                    # Freeze them!
                    self.visual_features.weight.requires_grad = False 
                    print(f"Successfully loaded and frozen 1280-dim image features from {img_path}")
                else:
                    print(f"Warning: Size Mismatch! File: {weights.shape[0]}, Model: {self.item_num+1}. Skipping images.")
            except Exception as e:
                print(f"Could not load visual features: {e}")


            # =================================================
            # FEATURE 2: TAGS (Multi-Hot -> Linear)
            # =================================================
            self.tag_features = None # Initialize as None first
            try:
                tag_matrix = np.load(tag_path)
                
                if tag_matrix.shape[0] == self.item_num + 1:
                    # FIRST: Define the attribute
                    self.num_unique_tags = tag_matrix.shape[1] 
                    
                    # SECOND: Define the Projection Layer using that attribute
                    self.tag_reduction = torch.nn.Linear(self.num_unique_tags, args.hidden_units)
                    
                    # THIRD: Add the Norm layer we discussed
                    self.tag_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                    
                    # FOURTH: Load the weights
                    self.tag_features = torch.nn.Embedding.from_pretrained(
                        torch.from_numpy(tag_matrix).float(), 
                        freeze=True
                    )
                    print(f"Successfully loaded {self.num_unique_tags} unique tags from {tag_path}")
                else:
                    print(f"Warning: Tag Size Mismatch! File: {tag_matrix.shape[0]}, Model: {self.item_num+1}")

            except Exception as e:
                print(f"Skipping Tags due to error: {e}")

            # =================================================

            self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
            self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

            self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
            self.attention_layers = torch.nn.ModuleList()
            self.forward_layernorms = torch.nn.ModuleList()
            self.forward_layers = torch.nn.ModuleList()

            self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

            for _ in range(args.num_blocks):
                new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                self.attention_layernorms.append(new_attn_layernorm)

                new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                                args.num_heads,
                                                                args.dropout_rate)
                self.attention_layers.append(new_attn_layer)

                new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                self.forward_layernorms.append(new_fwd_layernorm)

                new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
                self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        # 1. Base ID
        ids = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb(ids)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # --- NEW CODE: VISUAL PROJECTION ---
        
        # 2. Get Visual Embeddings (The "Content" part)
        # Look up the 1280 vectors
        visuals = self.visual_features(ids)
        
        # 3. Project Visuals: 1280 -> 50
        visuals_projected = self.visual_norm(self.dim_reduction_layer(visuals))
        # Logic: ID + (α * Visual)
        seqs = seqs + (self.alpha_visual * visuals_projected)
        # 4. Combine!
        # Option A: Addition (Most common) -> Input = ID + Image
        seqs = seqs + visuals_projected

        # 3. Add Tags (New Code)
        if self.tag_features is not None:
            # Get the Multi-Hot vector (Batch x Seq x NumTags)
            tag_vectors = self.tag_features(ids)
            # CHANGE: Replace tanh with LayerNorm
            tag_projected = self.tag_norm(self.tag_reduction(tag_vectors.float()))
            # Logic: (Current Sum) + (β * Tags)
            seqs = seqs + (self.alpha_tags * tag_projected)

        #finished new code-----------------------------
        
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
