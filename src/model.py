import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import random

from preprocess import *
# from sklearn.preprocessing import LabelEncoder
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int,
                    help="display a square of a given number")
parser.add_argument("--thre", type=int,
                    help="increase output verbosity")
parser.add_argument("--numin", type=int,
                    help="increase output verbosity")
parser.add_argument("--data", type=str,
                    help="increase output verbosity")
parser.add_argument("--ct_lambda", type=str,
                    help="increase output verbosity")
parser.add_argument("--cs_lambda", type=str,
                    help="increase output verbosity")
parser.add_argument("--att_lambda", type=str,
                    help="increase output verbosity")
args = parser.parse_args()

gpu = int(args.gpu)
num_interests = int(args.numin)
att_thre = 1 / float(args.thre)  # inverse
data_name = str(args.data)
ct_lambda = float(args.ct_lambda)
cs_lambda = float(args.cs_lambda)
att_lambda = float(args.att_lambda)
comi_ndcg = True

import pdb

TODO = -1
data_dic = {
    "book": [603668, 60367, 367982],
}


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()

        self.user_num, self.test_user_num, self.item_num = data_dic[data_name]

        self.batch_size = 1024
        self.sample_size = 5000
        self.seqence_length = 20
        self.embedding_dim = 64
        self.device = device
        self.embedding_layer = torch.nn.Embedding(self.item_num + 1, self.embedding_dim)

        # self.proposal_num = 20
        self.proposal_num = num_interests
        self.W1 = torch.nn.Parameter(data=torch.randn(256, self.embedding_dim), requires_grad=True)
        self.W1_2 = torch.nn.Parameter(data=torch.randn(self.proposal_num, 256), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3_2 = torch.nn.Parameter(data=torch.randn(self.seqence_length, self.embedding_dim), requires_grad=True)
        self.W5 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)

        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_cons = nn.Linear(self.embedding_dim, self.embedding_dim * self.seqence_length)

        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.recons_mse_loss = nn.MSELoss(reduce=False)

    def gen_dict(self, item_profile):
        # print(item_embedding_dict.shape)
        self.item_embedding_dict = torch.cat([torch.tensor([[0] * self.embedding_dim]).to(self.device),
                                              self.embedding_layer(item_profile.to(self.device))])
        # self.item_embedding_dict = F.normalize(self.item_embedding_dict, p=2, dim=1)

    def linear_layer(self, interests_matrix):
        interests_matrix = F.tanh(self.fc1(interests_matrix))
        return interests_matrix

    def create_interests(self, watch_movie, watch_movie_length, mode, t_att=0.02, t_cont=0.02, t_cons=0.5):
        import pdb
        watch_movie_length = watch_movie_length.cpu()
        import pdb
        dim0, dim1 = watch_movie.shape
        item_mask = (watch_movie == 0).reshape(dim0, -1)
        watch_movie = torch.reshape(watch_movie, (1, dim0 * dim1))
        watch_movie_embedding = self.item_embedding_dict[watch_movie]
        watch_movie_embedding = torch.reshape(watch_movie_embedding, (dim0, dim1, -1))

        proposals_weight = torch.matmul(self.W1_2,
                                        F.tanh(torch.matmul(self.W1, torch.transpose(watch_movie_embedding, 1, 2))))
        proposals_weight_logits = proposals_weight.masked_fill(item_mask.unsqueeze(1), -1e9)
        proposals_weight = torch.softmax(proposals_weight_logits, dim=2)
        watch_interests = torch.matmul(proposals_weight, torch.matmul(watch_movie_embedding, self.W2))

        if mode == 'test':
            return watch_interests
        elif mode == 'train':
            # re-attend
            product = torch.matmul(watch_interests, torch.transpose(watch_movie_embedding, 1, 2))
            product = product.masked_fill(item_mask.unsqueeze(1), -1e9)
            re_att = torch.softmax(product, dim=2)
            att_pred = F.log_softmax(proposals_weight_logits, dim=-1)
            loss_attend = -(re_att * att_pred).sum() / (re_att).sum()

            # re-contrast
            norm_watch_interests = F.normalize(watch_interests, p=2, dim=-1)
            norm_watch_movie_embedding = F.normalize(watch_movie_embedding, p=2, dim=-1)
            cos_sim = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_movie_embedding, 1, 2))
            if att_thre == -1:
                gate = np.repeat(1 / (watch_movie_length * 1.0), self.seqence_length, axis=0)
            else:
                gate = np.repeat(torch.FloatTensor([att_thre]).repeat(watch_movie_length.size(0)), self.seqence_length,
                                 axis=0)
            gate = torch.reshape(gate, (dim0, 1, self.seqence_length)).to(self.device)
            positive_weight_idx = (proposals_weight > gate) * 1  # value is 1 or 0
            mask_cos = cos_sim.masked_fill(item_mask.unsqueeze(1), -1e9)
            pos_cos = mask_cos.masked_fill(positive_weight_idx != 1, -1e9)
            import pdb
            # cons_pos = torch.sum(torch.exp(pos_cos / t_cont), dim=2)
            cons_pos = torch.exp(pos_cos / t_cont)
            cons_neg = torch.sum(torch.exp(mask_cos / t_cont), dim=2)

            in2in = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_interests, 1, 2))
            in2in = in2in.masked_fill(torch.eye(self.proposal_num).to(in2in.device).unsqueeze(0) == 1, -1e9)
            cons_neg = cons_neg + torch.sum(torch.exp(in2in / t_cont), dim=2)

            item_rolled = torch.roll(norm_watch_movie_embedding, 1, 0)
            in2i = torch.matmul(norm_watch_interests, torch.transpose(item_rolled, 1, 2))
            in2i_mask = torch.roll((watch_movie == 0).reshape(dim0, dim1), 1, 0)
            in2i = in2i.masked_fill(in2i_mask.unsqueeze(1), -1e9)
            cons_neg = cons_neg + torch.sum(torch.exp(in2i / t_cont), dim=2)

            cons_div = cons_pos / cons_neg.unsqueeze(-1)
            cons_div = cons_div.masked_fill(item_mask.unsqueeze(1), 1)
            cons_div = cons_div.masked_fill(positive_weight_idx != 1, 1)
            # loss_contrastive = -torch.log(cons_pos / cons_neg.unsqueeze(-1))
            loss_contrastive = -torch.log(cons_div)
            loss_contrastive = torch.mean(loss_contrastive)

            # re-construct
            recons_item = self.fc_cons(watch_interests)
            recons_item = recons_item.reshape([dim0 * self.proposal_num, dim1, -1])
            recons_weight = torch.matmul(self.W3_2,
                                         F.tanh(torch.matmul(self.W3, torch.transpose(recons_item, 1, 2))))
            recons_weight = recons_weight.reshape([dim0, self.proposal_num, dim1, dim1])
            recons_weight = recons_weight.masked_fill((watch_movie == 0).reshape(dim0, 1, 1, dim1), -1e9).reshape(
                [-1, dim1, dim1])
            recons_weight = torch.softmax(recons_weight, dim=-1)
            recons_item = torch.matmul(recons_weight, torch.matmul(recons_item, self.W5)).reshape(
                [dim0, self.proposal_num, dim1, -1])
            target_emb = watch_movie_embedding.unsqueeze(1).repeat(1, self.proposal_num, 1, 1)
            loss_construct = self.recons_mse_loss(recons_item, target_emb)
            loss_construct = loss_construct.masked_fill((positive_weight_idx == 0).unsqueeze(-1), 0.)
            loss_construct = loss_construct.masked_fill(item_mask.unsqueeze(-1).unsqueeze(1), 0.)
            loss_construct = torch.mean(loss_construct)
            return watch_interests, loss_attend, loss_contrastive, loss_construct

    def sampled_softmax(self, user_embedding, next_item, candidate_set, t=1):
        import pdb
        target_embedding = torch.sum(self.item_embedding_dict[next_item] * user_embedding, dim=1).view(
            len(user_embedding), 1)
        product = torch.matmul(user_embedding, torch.transpose(self.item_embedding_dict[candidate_set], 0, 1))
        # product = torch.cat([target_embedding, product], dim=1)
        loss = torch.exp(target_embedding / t) / (
                    torch.sum(torch.exp(product / t), dim=1, keepdim=True) + torch.exp(target_embedding))
        loss = torch.mean(-torch.log(loss))
        return loss

    # As stated in the paper, we follow the evaluation protocal of ComiRec----https://github.com/THUDM/ComiRec
    def eva(self, pre, ground_truth):
        hit20, recall20, NDCG20, hit50, recall50, NDCG50 = (0, 0, 0, 0, 0, 0)
        epsilon = 0.1 ** 10
        for i in range(len(ground_truth)):
            one_DCG20, one_recall20, IDCG20, one_hit20, one_DCG50, one_recall50, IDCG50, one_hit50 = (
            0, 0, 0, 0, 0, 0, 0, 0)
            top_20_item = pre[i][0:20].tolist()
            top_50_item = pre[i][0:50].tolist()
            positive_item = ground_truth[i]

            for pos, iid in enumerate(positive_item):
                if iid in top_20_item:
                    one_recall20 += 1
                    one_DCG20 += 1 / np.log2(pos + 2)
                if iid in top_50_item:
                    one_recall50 += 1
                    one_DCG50 += 1 / np.log2(pos + 2)

            if comi_ndcg:
                for pos in range(one_recall20):
                    IDCG20 += 1 / np.log2(pos + 2)
                for pos in range(one_recall50):
                    IDCG50 += 1 / np.log2(pos + 2)
            else:
                for pos in range(len(positive_item[:20])):
                    IDCG20 += 1 / np.log2(pos + 2)
                for pos in range(len(positive_item[:50])):
                    IDCG50 += 1 / np.log2(pos + 2)

            NDCG20 += one_DCG20 / max(IDCG20, epsilon)
            NDCG50 += one_DCG50 / max(IDCG50, epsilon)
            top_20_item = set(top_20_item)
            top_50_item = set(top_50_item)
            positive_item = set(positive_item)
            if len(top_20_item & positive_item) > 0:
                hit20 += 1
            if len(top_50_item & positive_item) > 0:
                hit50 += 1
            recall20 += len(top_20_item & positive_item) / max(len(positive_item), epsilon)
            recall50 += len(top_50_item & positive_item) / max(len(positive_item), epsilon)
            # F1 += 2 * precision * recall / max(precision + recall, epsilon)

        return hit20, recall20, NDCG20, hit50, recall50, NDCG50


if __name__ == '__main__':
    device = torch.device("cuda:{}".format(gpu))
    datapath = "../dataset/{}/".format(data_name)

    # load data
    ml_train, ml_test, ml_valid = pd.read_csv(datapath + '{}_train.csv'.format(data_name)), pd.read_csv(
        datapath + '{}_test.csv'.format(data_name)), pd.read_csv(datapath + '{}_valid.csv'.format(data_name))
    user_profile = list(set(ml_train['uid'].tolist() + ml_test['uid'].tolist() + ml_valid['uid'].tolist()))
    item_profile = list(set(ml_train['sid'].tolist() + ml_test['sid'].tolist() + ml_valid['sid'].tolist()))
    all_item = item_profile
    item_profile = torch.tensor(item_profile)
    import pdb
    import os
    import pickle

    train_path = "{}/{}".format(datapath, "train.pickle")
    test_path = "{}/{}".format(datapath, "test.pickle")
    if not os.path.exists(train_path):
        # all_item = item_profile
        train_set = gen_train_set(ml_train)
        # valid_set = gen_test_set(ml_valid)
        test_set = gen_test_set(ml_test)
        train_model_input, train_label = gen_model_input(train_set, 20)
        # valid_model_input, valid_label = gen_model_input(valid_set, 20)
        test_model_input, test_label = gen_model_input(test_set, 20)
        # print(train_label)
        with open(train_path, 'wb') as f:
            pickle.dump([train_model_input, train_label], f)
        with open(test_path, 'wb') as f:
            pickle.dump([test_model_input, test_label], f)

    else:
        with open(train_path, "rb") as f:
            train_model_input, train_label = pickle.load(f)
        with open(test_path, "rb") as f:
            test_model_input, test_label = pickle.load(f)

    import pdb

    import pickle

    # optimizer
    model = Model(device)
    model = model.to(device)
    lr = 0.003
    weight_decay = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # cross_entropy_loss = nn.CrossEntropyLoss()
    # contrastive_loss = nn.TripletMarginLoss(margin=1, p=2)
    # cosine_contrative_loss = nn.CosineEmbeddingLoss(0.5)

    record_length = len(train_model_input['user_id'])
    train_batch_num = (record_length // model.batch_size) + 1
    file_name = 'model_{}_{}_{}_{}_{}_{}.txt'.format(gpu, att_thre, data_name, ct_lambda, cs_lambda, att_lambda)
    epochs = 500
    best_sum = 0
    bias = np.arange(0, 1024 * num_interests, num_interests)
    bias = torch.tensor(bias).to(model.device)
    import time

    for epoch in range(epochs):
        sta_time = time.time()

        for i in range(train_batch_num):
            model.train()
            model.gen_dict(item_profile)
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, record_length)
            # calculate batch
            user_batch = train_model_input['user_id'][start:end]
            next_item = torch.tensor(train_model_input['movie_id'][start:end], dtype=torch.int64).to(device)
            watch_movie = torch.tensor(train_model_input['hist_movie_id'][start:end], dtype=torch.int64).to(device)
            watch_movie_length = torch.tensor(train_model_input['hist_len'][start:end], dtype=torch.int64).to(device)
            candidate_set = list(set(item_profile.tolist()) ^ set(next_item.tolist()))
            candidate_set = torch.tensor(random.sample(candidate_set, model.sample_size)).to(device)

            watch_interests, loss_attend, loss_contrastive, loss_construct = model.create_interests(watch_movie,
                                                                                                    watch_movie_length,
                                                                                                    mode='train')
            watch_interests = model.linear_layer(watch_interests)
            target_item_embedding = torch.unsqueeze(model.item_embedding_dict[next_item], dim=2)
            product = torch.matmul(watch_interests, target_item_embedding)
            import pdb

            watch_interests = watch_interests.reshape((len(watch_interests) * num_interests, 64))
            user_embedding_idx = torch.argmax(product, dim=1)
            length = len(user_embedding_idx)
            user_embedding_idx = torch.squeeze(user_embedding_idx, dim=1)
            user_embedding_idx = user_embedding_idx + bias[:length]
            user_embedding = torch.zeros((len(user_embedding_idx), model.embedding_dim)).to(model.device)
            user_embedding = watch_interests[user_embedding_idx, :]
            loss_base = model.sampled_softmax(user_embedding, next_item, candidate_set)

            loss = loss_base + att_lambda * loss_attend + ct_lambda * loss_contrastive + cs_lambda * loss_construct

            # print(loss_base)
            # print(loss_attend)
            # print(loss_contrastive)
            # print(loss_construct)
            with open("log/" + file_name, "a") as f:
                if i != 1 and i % 200 == 1:
                    f.write('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i) + ', ' +
                            'loss_base: ' + str(loss_base.item()) + ', ' +
                            'loss_attend: ' + str(loss_attend.item()) + ', ' +
                            'loss_contrastive: ' + str(loss_contrastive.item()) + ', ' +
                            'loss_construct: ' + str(loss_construct.item()) + '\n')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # evasteam 
            model.eval()
            if i == train_batch_num - 1:
                end_time = time.time()
                # print("train takes {}".format(end_time - sta_time))
                sta_time = time.time()
                if epoch == 0:
                    with open('test_result/' + file_name, "a") as f:
                        f.write('lr: ' + str(lr) + ' , ' + 'weight_decay: ' + str(weight_decay) + '\n')
                hit20, recall20, NDCG20, hit50, recall50, NDCG50 = (0, 0, 0, 0, 0, 0)
                with torch.no_grad():
                    test_batch_num = (len(test_model_input['user_id']) // 128) + 1
                    for j in range(test_batch_num):
                        start = j * 128
                        end = min((j + 1) * 128, len(test_model_input['user_id']))
                        # print(test_model_input['movie_id'][start:end])
                        next_item = test_model_input['movie_id'][start:end]
                        watch_movie = torch.tensor(test_model_input['hist_movie_id'][start:end],
                                                   dtype=torch.int64).to(device)
                        watch_movie_length = torch.tensor(test_model_input['hist_len'][start:end],
                                                          dtype=torch.int64).to(device)

                        user_embedding = model.create_interests(watch_movie, watch_movie_length, mode='test')
                        user_embedding = model.linear_layer(user_embedding)
                        product = torch.matmul(user_embedding, torch.transpose(model.item_embedding_dict[1:], 0, 1))
                        result, _ = torch.max(product, dim=1)
                        # print(result)
                        _, pre = torch.sort(result, descending=True)
                        pre += 1
                        result = model.eva(pre, next_item)
                        hit20 += result[0]
                        recall20 += result[1]
                        NDCG20 += result[2]
                        hit50 += result[3]
                        recall50 += result[4]
                        NDCG50 += result[5]
                    hit20 = hit20 / model.test_user_num
                    recall20 = recall20 / model.test_user_num
                    NDCG20 = NDCG20 / model.test_user_num
                    hit50 = hit50 / model.test_user_num
                    recall50 = recall50 / model.test_user_num
                    NDCG50 = NDCG50 / model.test_user_num

                    with open('test_result/' + file_name, "a") as f:
                        sum = 0
                        # torch.save(model.state_dict(), './best_model/best_baseline.pth')
                        sum = hit20 + recall20 + NDCG20 + hit50 + recall50 + NDCG50
                        if sum > best_sum:
                            best_sum = sum
                            f.write('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i) + ': ' + '\n')
                            print('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i) + ': ' + '\n')
                            # f.write('hit5: ' + str(hit5) + 'recall_5: ' + str(recall_5) + ' , ' + 'NDCG5: ' + str(NDCG5) + '\n')
                            # f.write('hit10: ' + str(hit10) + 'recall_10: ' + str(recall_10) + ' , ' + 'NDCG10: ' + str(NDCG10) + '\n')
                            f.write('hit20: ' + str(hit20) + 'recall_20: ' + str(recall20) + ' , ' + 'NDCG20: ' + str(
                                NDCG20) + '\n')
                            print('hit20: ' + str(hit20) + 'recall_20: ' + str(recall20) + ' , ' + 'NDCG20: ' + str(
                                NDCG20) + '\n')
                            f.write('hit50: ' + str(hit50) + 'recall_50: ' + str(recall50) + ' , ' + 'NDCG50: ' + str(
                                NDCG50) + '\n')
                            print('hit50: ' + str(hit50) + 'recall_50: ' + str(recall50) + ' , ' + 'NDCG50: ' + str(
                                NDCG50) + '\n')
                end_time = time.time()
                # print("eval takes {}".format(end_time - sta_time))
