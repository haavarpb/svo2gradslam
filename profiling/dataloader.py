from svo2gradslam.svo_dataset import SVOIterableDataset, sofa_filepath
from torch.utils.data import DataLoader
import pyinstrument
import tqdm


dataset = SVOIterableDataset(str(sofa_filepath()))

dl_iter = DataLoader(dataset, batch_size=10)

with pyinstrument.profile():
    for ds in tqdm.tqdm(dl_iter):
        pass
