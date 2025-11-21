import copy
import os
from copy import deepcopy
from os.path import join
from typing import Literal

import scipy.io
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_url, list_dir

from .base import BaseRealDataset


class StanfordDogs(BaseRealDataset):

    folder = "StanfordDogs"
    download_url_prefix = "http://vision.stanford.edu/aditya86/ImageNetDogs"

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):
        super().__init__()
        self.root = f"{data_root}/StanfordDogs"
        self.split = split
        self.download()

        split = self.load_split()

        self.images_folder = join(self.root, "Images")
        self.annotations_folder = join(self.root, "Annotation")
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + ".jpg", idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

        self.full_dataset = copy.deepcopy(self._flat_breed_images)

        self.class_names = [
            "Chihuaha",
            "Japanese Spaniel",
            "Maltese Dog",
            "Pekinese",
            "Shih-Tzu",
            "Blenheim Spaniel",
            "Papillon",
            "Toy Terrier",
            "Rhodesian Ridgeback",
            "Afghan Hound",
            "Basset Hound",
            "Beagle",
            "Bloodhound",
            "Bluetick",
            "Black-and-tan Coonhound",
            "Walker Hound",
            "English Foxhound",
            "Redbone",
            "Borzoi",
            "Irish Wolfhound",
            "Italian Greyhound",
            "Whippet",
            "Ibizian Hound",
            "Norwegian Elkhound",
            "Otterhound",
            "Saluki",
            "Scottish Deerhound",
            "Weimaraner",
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
            "Bedlington Terrier",
            "Border Terrier",
            "Kerry Blue Terrier",
            "Irish Terrier",
            "Norfolk Terrier",
            "Norwich Terrier",
            "Yorkshire Terrier",
            "Wirehaired Fox Terrier",
            "Lakeland Terrier",
            "Sealyham Terrier",
            "Airedale",
            "Cairn",
            "Australian Terrier",
            "Dandi Dinmont",
            "Boston Bull",
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
            "Tibetan Terrier",
            "Silky Terrier",
            "Soft-coated Wheaten Terrier",
            "West Highland White Terrier",
            "Lhasa",
            "Flat-coated Retriever",
            "Curly-coater Retriever",
            "Golden Retriever",
            "Labrador Retriever",
            "Chesapeake Bay Retriever",
            "German Short-haired Pointer",
            "Vizsla",
            "English Setter",
            "Irish Setter",
            "Gordon Setter",
            "Brittany",
            "Clumber",
            "English Springer Spaniel",
            "Welsh Springer Spaniel",
            "Cocker Spaniel",
            "Sussex Spaniel",
            "Irish Water Spaniel",
            "Kuvasz",
            "Schipperke",
            "Groenendael",
            "Malinois",
            "Briard",
            "Kelpie",
            "Komondor",
            "Old English Sheepdog",
            "Shetland Sheepdog",
            "Collie",
            "Border Collie",
            "Bouvier des Flandres",
            "Rottweiler",
            "German Shepard",
            "Doberman",
            "Miniature Pinscher",
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Appenzeller",
            "EntleBucher",
            "Boxer",
            "Bull Mastiff",
            "Tibetan Mastiff",
            "French Bulldog",
            "Great Dane",
            "Saint Bernard",
            "Eskimo Dog",
            "Malamute",
            "Siberian Husky",
            "Affenpinscher",
            "Basenji",
            "Pug",
            "Leonberg",
            "Newfoundland",
            "Great Pyrenees",
            "Samoyed",
            "Pomeranian",
            "Chow",
            "Keeshond",
            "Brabancon Griffon",
            "Pembroke",
            "Cardigan",
            "Toy Poodle",
            "Miniature Poodle",
            "Standard Poodle",
            "Mexican Hairless",
            "Dingo",
            "Dhole",
            "African Hunting Dog",
        ]

        self.num_classes = 120

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                transforms.Resize(res),
                (
                    transforms.CenterCrop(crop_res)
                    if crop_mode == "center"
                    else transforms.RandomCrop(crop_res)
                ),
                transforms.ToTensor(),
            ]
        )

        self.mean = torch.tensor(mean, device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        # if self.cropped:
        #     image = image.crop(self._flat_breed_annotations[index][1])
        #
        # if self.transform:
        #     image = self.transform(image)
        #
        # if self.target_transform:
        #     target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, "Images")) and os.path.exists(
            join(self.root, "Annotation")
        ):
            if (
                len(os.listdir(join(self.root, "Images")))
                == len(os.listdir(join(self.root, "Annotation")))
                == 120
            ):
                print("Files already downloaded and verified")
                return

        for filename in ["images", "annotation", "lists"]:
            tar_filename = filename + ".tar"
            url = self.download_url_prefix + "/" + tar_filename
            download_url(url, self.root, tar_filename, None)
            print("Extracting downloaded file: " + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), "r") as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.split == "train":
            split = scipy.io.loadmat(join(self.root, "train_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "train_list.mat"))["labels"]
        else:
            split = scipy.io.loadmat(join(self.root, "test_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "test_list.mat"))["labels"]

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print(
            "%d samples spanning %d classes (avg %f per class)"
            % (
                len(self._flat_breed_images),
                len(counts.keys()),
                float(len(self._flat_breed_images)) / float(len(counts.keys())),
            )
        )

        return counts

    def filter_dataset(self, class_idx: int):
        self._flat_breed_images = [d for d in self.full_dataset if d[1] == class_idx]

    def get_single_class(self, cls: int) -> Tensor:
        self._flat_breed_images = [d for d in self.full_dataset if d[1] == cls]
        num_samples = len(self._flat_breed_images)
        loader = DataLoader(self, batch_size=num_samples, num_workers=8)
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls}...")
        for x, y in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")
        self._flat_breed_images = deepcopy(self.full_dataset)

        return images
