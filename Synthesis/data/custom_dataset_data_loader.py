import torch.utils.data
from data.synthesis_dataset import SynthesisDataset


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

def CreateDataset(opt):
    dataset = None
    dataset = SynthesisDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader