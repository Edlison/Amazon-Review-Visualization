# @Author  : Edlison
# @Date    : 4/16/23 12:49
import torch
import numpy as np
import json
import requests as r
from torch_geometric.utils import degree


class Amazon:
    def __init__(self):
        self.also_buy, self.also_view, self.category, self.rating, self.item, self.item_name, self.category_name \
        = self._load_data()

    def _load_data(self):
        data_file = './data/Video_Games/thre100/data_dict.pt'
        mapping_file = './data/Video_Games/thre100/mapping_dict.pt'

        data_dict = torch.load(data_file)
        mapping_dict = torch.load(mapping_file)

        also_buy = torch.permute(torch.tensor(data_dict['also_buy']), [1, 0])
        also_view = torch.permute(torch.tensor(data_dict['also_view']), [1, 0])
        category = data_dict['category']  # 1:N, [1311, 2]
        rating = data_dict['rating']  # 1:N, [8804,]
        item = data_dict['item']  # [8804,]
        itemname = mapping_dict['itemname']  # [439]
        categoryname = mapping_dict['category']
        category = self._parse_cat(category)

        return also_buy, also_view, category, rating, item, itemname, categoryname

    def most_also(self, limits=10, reverse=True):
        _, buy_tgt = self.also_buy
        _, view_tgt = self.also_view
        deg_buy = degree(buy_tgt)
        deg_view = degree(view_tgt)
        idx_buy = np.argsort(deg_buy.numpy())
        idx_view = np.argsort(deg_view.numpy())
        if reverse:
            idx_buy = idx_buy[::-1][:limits]
            idx_view = idx_view[::-1][:limits]
        else:
            idx_buy = idx_buy[:limits]
            idx_view = idx_view[:limits]
        return idx_buy, idx_view

    def highest_rating(self, limits, reverse=True):
        res = []
        for i in range(len(self.item_name)):
            rate_idx = np.argwhere(self.item == i)
            rate = np.mean([self.rating[idx] for idx in rate_idx])
            res.append(rate)
        idx_rating = np.argsort(res)
        if reverse:
            idx_item = idx_rating[::-1][:limits]
        else:
            idx_item = idx_rating[:limits]
        rate_item = [res[i] for i in idx_item]
        return idx_item, rate_item

    def index2info(self, item_index):
        name, cat = self._get_name(item_index), self._get_cat(item_index)
        res = []
        for n, c in zip(name, cat):
            each = "{} # Categories: [{}]".format(n, ','.join(c))
            res.append(each)
        return res

    def _get_name(self, item_index):
        return [self.item_name[index] for index in item_index]

    def _get_cat(self, item_index):
        res = []
        for i in item_index:
            cat_idx = self.category[int(i)]
            cat_name = [self.category_name[idx] for idx in cat_idx]
            res.append(cat_name)
        return res

    def _parse_cat(self, category):
        src = category[:, 0]
        tgt = category[:, 1]
        res = {}
        for i in range(max(src)+1):
            res[i] = []
        for i, j in zip(src, tgt):
            res[i].append(j)
        return res


class AmazonDataset:
    def __init__(self, relation='also_buy'):
        self.x, self.edge_dict = self.load_data()
        self.edge_index = self.edge_dict[relation]
        self.y = self.get_label()
        self.num_node_features = self.x.shape[1]
        self.num_classes = 5
        self.train_ratio = 0.8
        self.train_mask, self.test_mask = self.gen_mask()

    def load_data(self):
        emb_file = './data/Video_Games/thre100/emb.pt'
        emb = torch.load(emb_file)
        emb = torch.tensor(emb)

        data_file = './data/Video_Games/thre100/data_dict.pt'
        data_dict = torch.load(data_file)
        also_buy = torch.permute(torch.tensor(data_dict['also_buy']), [1, 0])
        also_view = torch.permute(torch.tensor(data_dict['also_view']), [1, 0])
        edge_dict = {'also_buy': also_buy, 'also_view': also_view}

        return emb, edge_dict

    def get_label(self):
        data_file = './data/Video_Games/thre100/data_dict.pt'
        data_dict = torch.load(data_file)

        category = data_dict['category']
        item_idx = category[:, 0]
        cat4_idx = np.argwhere(category[:, 1] == 4)
        cat3_idx = np.argwhere(category[:, 1] == 3)
        cat2_idx = np.argwhere(category[:, 1] == 2)
        cat1_idx = np.argwhere(category[:, 1] == 1)
        cat0_idx = np.argwhere(category[:, 1] == 0)
        cat4_item = category[cat4_idx, 0]
        cat3_item = category[cat3_idx, 0]
        cat2_item = category[cat2_idx, 0]
        cat1_item = category[cat1_idx, 0]
        cat0_item = category[cat0_idx, 0]
        uni_cat4 = cat4_item
        uni_cat3 = [x for x in cat3_item if x not in cat4_item]
        uni_cat2 = [x for x in cat2_item if x not in cat4_item and x not in cat3_item]
        uni_cat1 = [x for x in cat1_item if x not in cat4_item and x not in cat3_item and x not in cat2_item]
        uni_cat0 = [x for x in cat0_item if
                    x not in cat4_item and x not in cat3_item and x not in cat2_item and x not in cat1_item]

        label = torch.zeros(data_dict['num_items'], dtype=torch.long)
        for i in uni_cat4:
            label[i] = 4
        for i in uni_cat3:
            label[i] = 3
        for i in uni_cat2:
            label[i] = 2
        for i in uni_cat1:
            label[i] = 1
        return label

    def gen_mask(self):
        size = self.x.shape[0]
        train_mask = torch.zeros(size, dtype=torch.bool)
        test_mask = torch.zeros(size, dtype=torch.bool)
        train_index = torch.arange(0, self.train_ratio * size, dtype=torch.long)
        test_index = torch.arange(self.train_ratio * size, size, dtype=torch.long)

        train_mask[train_index] = True
        test_mask[test_index] = True

        return train_mask, test_mask