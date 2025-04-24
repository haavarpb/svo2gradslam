import pyinstrument
import tqdm

from svo2gradslam.svo_dataset import SVOIterableDataset, sofa_filepath

dataset = SVOIterableDataset(str(sofa_filepath()))

with pyinstrument.profile():
    for sample in tqdm.tqdm(dataset):
        sample
