import tqdm
from torch.profiler import ProfilerActivity, profile

from svo2gradslam.svo_dataset import SVOIterableDataset, sofa_filepath

device = 'cuda'

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
dataset = SVOIterableDataset(str(sofa_filepath()), end=10)

with profile(activities=activities) as prof:
    for sample in tqdm.tqdm(dataset):
        sample

prof.export_chrome_trace("trace.json")