import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
        
        # This represents 'd_i' from the paper
        self.delta_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
    
        # Paper uses a small delta_max (e.g., 0.1 to 0.2)
        self.delta_max = getattr(args, 'delta_max', 0.1) 
        
        # FIX: Handle potential absolute paths in args.dataset
        if os.path.isabs(args.dataset):
            dataset_subdir = os.path.dirname(args.dataset)
            # Check if we are already inside a structure that has data/
            if dataset_subdir.endswith('data_70_30') or dataset_subdir.endswith('data_loo'):
                 data_root = dataset_subdir 
            else:
                 # Fallback if path is weird
                 data_root = os.path.join(dataset_subdir)
        else:
             dataset_subdir = os.path.dirname(args.dataset)
             if os.path.exists(os.path.join('data', dataset_subdir)):
                data_root = os.path.join('data', dataset_subdir)
             else:
                data_root = os.path.join('..', 'data', dataset_subdir)

        print(f"--> [Model] Loading features from root: {data_root}")
        
        self.use_visual = args.use_visual
        self.use_tags = args.use_tags
        
        # 1. VISUAL FEATURES (REMAINING AS FROZEN CONTENT c_i)
        if self.use_visual:
            img_filename = 'pretrained_group_emb_1280.npy' if 'group' in args.dataset else 'pretrained_item_emb_1280.npy'
            img_path = os.path.join(data_root, img_filename)
            self.dim_reduction_layer = torch.nn.Linear(1280, args.hidden_units)
            self.visual_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.visual_features = torch.nn.Embedding(self.item_num+1, 1280, padding_idx=0)
            
            # Safe loading
            if os.path.exists(img_path):
                weights = np.load(img_path)
                self.visual_features.weight.data.copy_(torch.from_numpy(weights))
            else:
                print(f"Warning: Visual embeddings not found at {img_path}. Initializing random.")
                
            self.visual_features.weight.requires_grad = False

        # 2. TAG FEATURES (REMAINING AS FROZEN CONTENT c_i)
        if self.use_tags:
            tag_filename = 'pretrained_group_tag_emb.npy' if 'group' in args.dataset else 'pretrained_tag_emb.npy'
            tag_path = os.path.join(data_root, tag_filename)
            
            if os.path.exists(tag_path):
                tag_matrix = np.load(tag_path)
                self.tag_reduction = torch.nn.Linear(tag_matrix.shape[1], args.hidden_units)
                self.tag_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                self.tag_features = torch.nn.Embedding.from_pretrained(torch.from_numpy(tag_matrix).float(), freeze=True)
            else:
                print(f"Warning: Tag embeddings not found at {tag_path}. Disabling tags.")
                self.use_tags = False
        
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
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
    
    def get_item_vector(self, ids):
        # Implementation of Section 3.3: e_i = c_i + d_i
        
        # --- FIX: Correct Initialization of c_i ---
        # Instead of just taking ids.shape[0] (which breaks for 2D inputs),
        # we create a tensor of zeros matching the full shape of ids + hidden_dim.
        # This handles both (Batch, Sequence) and (ItemNum) shapes.
        shape = ids.shape + (self.delta_emb.embedding_dim,)
        c_i = torch.zeros(shape, device=self.dev)

        if self.use_visual:
            visuals = self.visual_features(ids)
            c_i += self.visual_norm(self.dim_reduction_layer(visuals))
        if self.use_tags:
            tags = self.tag_features(ids)
            c_i += self.tag_norm(self.tag_reduction(tags))
        
        # Section 3.3: Fix their norm (unit norm)
        c_i = torch.nn.functional.normalize(c_i, p=2, dim=-1)

        # 2. Get Trainable Delta (d_i)
        d_i = self.delta_emb(ids)
        
        # 3. Final Representation: e_i = c_i + d_i
        return c_i + d_i

    def log2feats(self, log_seqs):
        ids = torch.LongTensor(log_seqs).to(self.dev)
        # Use our new adjustment logic
        seqs = self.get_item_vector(ids)
        
        # Standard SASRec scaling and positional encoding
        seqs *= self.delta_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
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

        # Updated to use get_item_vector for pos/neg items as well
        # This ensures they get the content+delta embedding
        pos_embs = self.get_item_vector(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.get_item_vector(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        # Updated to use get_item_vector
        item_embs = self.get_item_vector(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)'/