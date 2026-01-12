import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
import time
import torch
import argparse
import numpy as np # Added explicit numpy import just in case

from model_lg import SASRec
from utils_lg import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.7, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--use_visual', default=True, type=str2bool)
parser.add_argument('--use_tags', default=True, type=str2bool)
parser.add_argument('--delta_max', default=0.1, type=float)

args = parser.parse_args()

# --- SYNCED FOLDER SETUP ---
dataset_name = os.path.basename(args.dataset) 
model_folder = f"{dataset_name}_{args.train_dir}"

if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

with open(os.path.join(model_folder, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

log_file = open(os.path.join(model_folder, 'log.txt'), 'w')
log_file.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')

if __name__ == '__main__':
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    # FIX: Handle absolute paths correctly for data_root
    if os.path.isabs(args.dataset):
        data_root = os.path.dirname(args.dataset)
    else:
        dataset_subdir = os.path.dirname(args.dataset) 
        data_root = os.path.join('data', dataset_subdir)

    if 'group' in args.dataset:
        map_file = os.path.join(data_root, 'group_maps.pkl')
    else:
        map_file = os.path.join(data_root, 'item_maps.pkl')

    with open(map_file, 'rb') as f:
        maps = pickle.load(f)
        itemnum = len(maps[1])
    
    print(f'Total Catalog Items: {itemnum}')
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.delta_emb.weight.data[0, :] = 0

    model.train()
    epoch_start_idx = 1
    
    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        exit()

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-5)

    patience = 5
    patience_counter = 0 
    best_val_ndcg, best_val_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()

    f = open(os.path.join(model_folder, 'log.txt'), 'w')

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            for param in model.delta_emb.parameters(): 
                loss += args.l2_emb * torch.sum(param ** 2)
            
            loss.backward()
            adam_optimizer.step()

            with torch.no_grad():
                delta_weights = model.delta_emb.weight.data
                norms = delta_weights.norm(p=2, dim=1, keepdim=True)
                delta_weights.mul_(torch.clamp(args.delta_max / (norms + 1e-8), max=1.0))
                delta_weights[0] = 0

            if step % 10 == 0:
                print(f"loss in epoch {epoch} iteration {step}: {loss.item()}")

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating...')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            
            print('epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            
            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()

            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                patience_counter = 0
                fname = 'SASRec.epoch={}.pth'.format(epoch)
                torch.save(model.state_dict(), os.path.join(model_folder, fname))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping at epoch {epoch}")
                    break

            t0 = time.time()
            model.train()

    f.close()
    sampler.close()