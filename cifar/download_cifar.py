# DOWNLOAD STARTING KIT

import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive

url = "https://membershipinference.blob.core.windows.net/mico/cifar10.zip?si=cifar10&spr=https&sv=2021-06-08&sr=b&sig=d7lmXZ7SFF4ZWusbueK%2Bnssm%2BsskRXsovy2%2F5RBzylg%3D" 
filename = "cifar10_kit.zip"
md5 = "c615b172eb42aac01f3a0737540944b1"

# WARNING: this will download and extract a 2.1GiB file, if not already present. Please save the file and avoid re-downloading it.
download_and_extract_archive(url=url, download_root=os.curdir, extract_root=None, filename=filename, md5=md5, remove_finished=False)


# DOWNLOAD DATASET
from mico_competition import load_cifar10
full_dataset = load_cifar10(dataset_dir="./cifar10_data/", download=True)
