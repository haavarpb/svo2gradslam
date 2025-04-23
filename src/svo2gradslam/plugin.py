from svo2gradslam.svo_dataset import SVOIterableDataset


def svo_dataset(config_dict, basedir, sequence, **kwargs):
    if "svo_file" not in config_dict.keys():
        raise ValueError("SVO dataset requires svo_file")
    svo_file = config_dict["svo_file"]
    return SVOIterableDataset(svo_file, kwargs=kwargs)