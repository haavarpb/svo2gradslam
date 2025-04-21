from svo2gradslam.svo_dataset import SVOIterableDataset, sofa_filepath
import pyinstrument
import tqdm

dataset = SVOIterableDataset(str(sofa_filepath()))

with pyinstrument.profile():
    for sample in tqdm.tqdm(dataset):
        sample
