import warnings
import numpy as np
import pandas as pd
import random
import torch.utils.data
from sklearn.preprocessing import OrdinalEncoder
from transformers import BertTokenizer, BertModel
import torch
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, log_loss
from torch.utils.data import DataLoader
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel

warnings.filterwarnings('ignore')

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def text_transfrom(review_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    indexed_tokens = tokenizer.encode(review_text, add_special_tokens = True)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(indexed_tokens)
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = list(sentence_embedding)
    return sentence_embedding

class YelpBusinessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tiny_set, tiny_size, location, sep=',', engine='c', header='infer'):
        # data = pd.read_csv("yelp.csv", sep=',', engine='c', header='infer')
        data1 = pd.read_csv("yelp_process-1.csv", sep=',', engine='c', header=None)
        data2 = pd.read_csv("yelp_process-2.csv", sep=',', engine='c', header=None)
        data3 = pd.read_csv("yelp_process-3.csv", sep=',', engine='c', header=None)
        data4 = pd.read_csv("yelp_process-4.csv", sep=',', engine='c', header=None)
        data = pd.concat([data1, data2, data3, data4], ignore_index=True)
        data = data.to_numpy()
        if tiny_set:
            self.text = np.concatenate([data[:int(tiny_size / 2), 5:].astype(np.float32),
                                        data[int(120000 - tiny_size / 2):, 5:].astype(np.float32)], axis=0)
        else:
            self.text = data[:, 5:].astype(np.float32)
        data = pd.read_csv("yelp.csv", sep=',', engine='c', header='infer')
        data = data[['business_id', 'city', 'state', 'user_id', 'stars_x']]
        enc = OrdinalEncoder()
        mapping1 = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 1, 5.0: 1}
        data['stars_x'] = data['stars_x'].map(mapping1)
        data = data.to_numpy()
        data1 = enc.fit_transform(data)
        # print(data1)
        if tiny_set:
            data = np.concatenate(
                [data1[:int(tiny_size / 2), :].astype(np.int), data1[int(120000 - tiny_size / 2):, :].astype(np.int)],
                axis=0)
        else:
            data = data1
        # data1 = enc.fit_transform(data[:60000, 0:4])
        # data1 = enc.fit_transform(data)
        # 0:'business_id', 1:'city', 2:'state', 3:'user_id', 4:'stars_x', 5-772:'text',
        # data_process = np.concatenate([enc.fit_transform(data[:, 1:5]), data[:, -1].reshape(-1, 1), text_array], axis=1)
        if location:
            self.items_b = data1[:, 0:3].astype(np.int)
        else:
            self.items_b = data1[:, 0].reshape(-1, 1).astype(np.int)
        self.items_u = data1[:, 3].reshape(-1, 1).astype(np.int)
        # self.items = np.concatenate([data[:, 0].reshape(-1, 1).astype(np.int), data[:, 2].reshape(-1, 1).astype(np.int), data[:, 3].reshape(-1, 1).astype(np.int)], axis=1)
        self.targets = data[:, 4].astype(np.float32)
        self.field_dims1 = np.max(self.items_b, axis=0) + 1
        # self.field_dims2 = (np.max(self.items_u, axis=0) + 1).reshape(1,1)
        self.field_dims2 = np.max(self.items_u, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        # print(self.items)
        # print(self.targets)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        # return [self.items[index], self.text[index]], self.targets[index]
        # print(self.items[index], self.targets[index])
        return [self.items_b[index], self.items_u[index], self.text[index]], self.targets[index]

def get_dataset(name, path, tiny_set, tiny_size, location):
    if name == 'yelp':
        return YelpBusinessDataset(path, tiny_set, tiny_size, location)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset, add_text):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims1 = dataset.field_dims1
    field_dims2 = dataset.field_dims2
    if name == 'fm':
        return FactorizationMachineModel(field_dims1, field_dims2, embed_dim=16, add_text=add_text)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims1, field_dims2, embed_dim=2, mlp_dims=(2,1), dropout=0.4, add_text=add_text)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims1, field_dims2, embed_dim=2, num_layers=1, mlp_dims=([2]), dropout=0.5, add_text=add_text)
    else:
        raise ValueError('unknown model name: ' + name)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, epoch, log_interval=100, adv=False, domain_similar=False, add_text=False):
    model.train()
    # print("--------------------")
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        # print(fields)
        f_b = fields[0]
        f_u = fields[1]
        f_t = fields[2]
        f_b, f_u, f_t, target = f_b.to(device), f_u.to(device), f_t.to(device), target.to(device)
        optimizer.zero_grad()
        _, loss, loss_sim = model(f_b, f_u, f_t, True, epoch, criterion, target.float(),adv=adv,domain_similar=domain_similar, add_text=add_text)
        # loss_ = criterion(y, target.float())
        # loss = loss_ + 0.1 * loss
        # model.zero_grad()
        loss = loss + 0.05 * loss_sim
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device, add_text=False):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            f_b = fields[0]
            f_u = fields[1]
            f_t = fields[2]
            f_b, f_u, f_t, target = f_b.to(device), f_u.to(device), f_t.to(device), target.to(device)
            y,_,_ = model(f_b, f_u, f_t, False, epoch=-1, loss_fct=None,label=target.float(), add_text=add_text)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            # print(targets,predicts)
    # return roc_auc_score(targets, predicts), precision_score(targets, np.round(predicts)), log_loss(targets, predicts)
    return roc_auc_score(targets, predicts), f1_score(targets,np.round(predicts)), precision_score(targets, np.round(predicts)), log_loss(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         adv,
         domain_similar,
         add_text,
         tiny_set,
         tiny_size,
         location
         ):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, tiny_set, tiny_size, location)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    print(train_length,valid_length,test_length)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length),)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size,num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0)
    model = get_model(model_name, dataset, add_text).to(device)
    # for ind, i in model.state_dict().items():
    #     print(ind, i.shape)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    # print('----------------------------')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device, epoch,adv=adv,domain_similar=domain_similar, add_text=add_text)
        auc, f1, pr, log = test(model, valid_data_loader, device, add_text=add_text)
        print('epoch:', epoch_i, 'validation: auc:', auc, 'validation: f1:', f1, '\nvalidation: precision:', pr, 'validation: logloss:', log)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc, f1, pr, log = test(model, test_data_loader, device, add_text=add_text)
    print(f'test auc: {auc}, test f1: {f1}, test precision: {pr}, test logloss: {log}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='yelp')
    parser.add_argument('--dataset_path', default='yelp', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='dfm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--rnum', type=int, default=5)  # 5
    parser.add_argument('--adv', type=bool, default=True)
    parser.add_argument('--domain_similar', type=bool, default=False)
    parser.add_argument('--add_text', type=bool, default=True)
    parser.add_argument('--tiny_set', type=bool, default=False)
    parser.add_argument('--tiny_size', type=int, default=60000)
    parser.add_argument('--location', type=bool, default=True)
    args = parser.parse_args()
    set_random_seed(43)
    for i in range(args.rnum):
        main(args.dataset_name,
             args.dataset_path,
             args.model_name,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.device,
             args.save_dir,
             args.adv,
             args.domain_similar,
             args.add_text,
             args.tiny_set,
             args.tiny_size,
             args.location)