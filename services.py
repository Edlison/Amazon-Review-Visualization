# @Author  : Edlison
# @Date    : 3/13/23 19:18
import dgl
from pyvis.network import Network
import networkx as nx
from dataset import Amazon, AmazonDataset
import torch
import json
import requests as r
from model import NetAmazon_GAT


def draw():
    dataset = dgl.data.CoraGraphDataset()
    g = Network(height=800, width=800, notebook=True)
    netxG = nx.Graph(dataset[0].to_networkx())
    mapping = {i: i for i in range(netxG.size())}  # Setting mapping for the relabeling
    netxH = nx.relabel_nodes(netxG, mapping)  # relabeling nodes
    g.from_nx(netxH)
    g.show('./templates/graph.html')


def most(limits=10, type='buy', reverse=True):
    ds = Amazon()
    most_buy, most_view = ds.most_also(limits=limits, reverse=reverse)
    info_buy = ds.index2info(most_buy)
    info_view = ds.index2info(most_view)

    res = {'buy': info_buy, 'view': info_view}
    return res[type]


def rating(limits=10, reverse=True):
    ds = Amazon()
    item_idx, item_rate = ds.highest_rating(limits=limits, reverse=reverse)
    info_item = ds.index2info(item_idx)
    res = []
    for i, r in zip(info_item, item_rate):
        each = "{} # Rating: {:.2f}".format(i, r)
        res.append(each)

    return {'rate': res}


def get_emb(text):
    url = 'https://api.openai.com/v1/embeddings'
    auth = 'Bearer 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'
    data = {"model": "text-embedding-ada-002", "input": text}
    resp = r.post(url, headers={"Content-Type": "application/json", "Authorization": auth}, data=json.dumps(data))
    data = json.loads(resp.text)
    emb = data['data'][0]['embedding']
    file = './data/Video_Games/thre100/emb_single.pt'
    emb = torch.tensor(emb)
    emb = emb.view(1, -1)
    torch.save(emb, file)
    return emb


def inference(text, from_openai=True):
    if from_openai:
        print('get emb from openai...')
        emb = get_emb(text)
    else:
        print('get emb from cache...')
        file = './data/Video_Games/thre100/emb_single.pt'
        emb = torch.load(file)
    model_file = './data/Video_Games/thre100/model.pt'
    print('load model')
    amazonds = AmazonDataset()
    ds = Amazon()
    model = NetAmazon_GAT(amazonds.num_node_features, amazonds.num_classes)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print('infer...')
    edge_index = torch.tensor([[0], [0]])
    pred = model(emb, edge_index).argmax(dim=1)
    pred_name = ds.category_name[pred]
    print('pred name: ', pred_name)

    return pred_name


if __name__ == '__main__':
    draw()
