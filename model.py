# @Author  : Edlison
# @Date    : 4/18/23 16:37
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from dataset import AmazonDataset

cuda = True if torch.cuda.is_available() else False


class NetAmazon_GAT(torch.nn.Module):
    # todo add layers, without dropout, change activator
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=16)
        self.gat2 = GATConv(256, 8, heads=8)
        self.gat3 = GATConv(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.gat3(h, edge_index)

        return F.log_softmax(h, dim=1)


def train(iterations=100, lr=0.005, reg=5e-4):
    dataset = AmazonDataset()
    model = NetAmazon_GAT(dataset.num_node_features, dataset.num_classes)
    data = dataset
    x, edge_index = data.x, data.edge_index
    if cuda:
        device = 1
        x.cuda(device)
        edge_index(device)
        model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

    model.eval()
    model_path = './data/Video_Games/thre100/model.pt'
    torch.save(model.state_dict(), model_path)

    pred = model(x, edge_index).argmax(dim=1)
    cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = cor / data.test_mask.sum()
    print('Final acc: {:.4f}. Model saved in {}'.format(acc.item(), model_path))


if __name__ == '__main__':
    train(iterations=100)
