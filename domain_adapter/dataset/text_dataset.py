from torch.utils.data.dataset import Dataset

class TextDataset(Dataset):

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        #self.labels = labels

        self.inputs = {key: val for key, val in encodings.items()}
        #self.inputs['labels'] = labels

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        #item['labels'] = self.labels[index]

        import torch
        item2 = {key: val[index] for key, val in self.inputs.items()}
        for v1, v2 in zip(item.items(), item2.items()):
            assert v1[0] == v2[0]
            assert torch.all(torch.eq(v1[1], v2[1]))

        return item

    def __len__(self):
        #return len(self.labels)
        return len(self.encodings)