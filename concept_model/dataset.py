import pathlib

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive

URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
TGZ_MD5 = "97eceeb196236b17998738112f37df78"
ROOT = pathlib.Path(__file__).parent.resolve() / "data"


class CUBImageToAttributes(Dataset):
    def __init__(self, train: bool, download: bool = True):
        super().__init__()

        if download and not (ROOT / "CUB_200_2011").exists():
            download_and_extract_archive(
                url=URL,
                download_root=str(ROOT),
                filename="CUB_200_2011.tgz",
                md5=TGZ_MD5,
            )

        train_test_split = load_train_test_split()
        self.image_attributes = load_image_attribute_labels(train, train_test_split)
        self.image_paths = load_image_paths(train, train_test_split)

    def __len__(self):
        return len(self.image_attributes)

    def __getitem__(self, idx: int):
        image_path = ROOT / "CUB_200_2011" / "images" / self.image_paths[idx]

        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img = preprocess(pil_loader(str(image_path)))

        target = torch.from_numpy(self.image_attributes[idx])

        return img, target


class CUBAttributesToClass(Dataset):
    def __init__(self, train: bool, download: bool = True):
        super().__init__()

        if download and not (ROOT / "CUB_200_2011").exists():
            download_and_extract_archive(
                url=URL,
                download_root=str(ROOT),
                filename="CUB_200_2011.tgz",
                md5=TGZ_MD5,
            )

        train_test_split = load_train_test_split()
        self.image_class = load_image_class_labels(train, train_test_split)
        self.image_attributes = load_image_attribute_labels(train, train_test_split)

    def __len__(self):
        return len(self.image_class)

    def __getitem__(self, idx: int):
        attributes = torch.from_numpy(self.image_attributes[idx])
        target = self.image_class[idx] - 1

        return attributes, target


def load_train_test_split():
    filepath = ROOT / "CUB_200_2011" / "train_test_split.txt"
    return np.loadtxt(filepath, usecols=1, dtype=np.int_)


def load_image_attribute_labels(
    train: bool, train_test_split: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    filepath = ROOT / "CUB_200_2011" / "attributes" / "image_attribute_labels.txt"
    image_attributes = np.loadtxt(filepath, usecols=(0, 2), dtype=np.int_)
    grouped = np.stack(
        np.split(
            image_attributes[:, 1],
            np.unique(image_attributes[:, 0], return_index=True)[1][1:],
        )
    )
    if train:
        return grouped[np.nonzero(train_test_split == 1)]
    else:
        return grouped[np.nonzero(train_test_split == 0)]


def load_image_paths(
    train: bool, train_test_split: npt.NDArray[np.int_]
) -> tuple[str, ...]:
    filepath = ROOT / "CUB_200_2011" / "images.txt"
    with open(filepath, encoding="utf-8") as file:
        image_paths = tuple(line.split()[1] for line in file.readlines())
    if train:
        return tuple(image_paths[i] for i in np.nonzero(train_test_split == 1)[0])
    else:
        return tuple(image_paths[i] for i in np.nonzero(train_test_split == 0)[0])


def load_image_class_labels(
    train: bool, train_test_split: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    filepath = ROOT / "CUB_200_2011" / "image_class_labels.txt"
    image_class = np.loadtxt(filepath, usecols=1, dtype=np.int_)
    if train:
        return image_class[np.nonzero(train_test_split == 1)]
    else:
        return image_class[np.nonzero(train_test_split == 0)]
