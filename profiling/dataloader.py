import pyinstrument
import tqdm
from torch.utils.data import DataLoader

from svo2gradslam.svo_dataset import SVOIterableDataset, collate_sequence, sofa_filepath

dataset = SVOIterableDataset(str(sofa_filepath()))

dl_iter = DataLoader(dataset, batch_size=10, collate_fn=collate_sequence)

with pyinstrument.profile():
    for ds in tqdm.tqdm(dl_iter):
        pass
