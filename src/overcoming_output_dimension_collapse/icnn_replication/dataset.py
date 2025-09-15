from __future__ import annotations

from pathlib import Path

import numpy as np
from bdpy.dataform import DecodedFeatures, Features
from PIL import Image
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        layer_path_names: list[str],
        stimulus_names: list[str] | None = None,
        transform=None,
    ):
        self.features_store = Features(Path(root_path).as_posix())
        self.layer_path_names = layer_path_names
        if stimulus_names is None:
            stimulus_names = self.features_store.labels
        self.stimulus_names = stimulus_names
        self.transform = transform

    def __len__(self):
        return len(self.stimulus_names)

    def __getitem__(self, index: int):
        stimulus_name = self.stimulus_names[index]
        features = {}
        for layer_path_name in self.layer_path_names:
            feature = self.features_store.get(
                layer=layer_path_name, label=stimulus_name
            )
            feature = feature[0]  # NOTE: remove batch axis
            features[layer_path_name] = feature
        if self.transform is not None:
            features = self.transform(features)
        return features


class DecodedFeaturesDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        layer_path_names: list[str],
        subject_id: str,
        roi: str,
        stimulus_names: list[str] | None = None,
        transform=None,
    ):
        self.decoded_features_store = DecodedFeatures(Path(root_path).as_posix())
        self.layer_path_names = layer_path_names
        self.subject_id = subject_id
        self.roi = roi
        if stimulus_names is None:
            stimulus_names = self.decoded_features_store.labels
            assert stimulus_names is not None
        self.stimulus_names = stimulus_names
        self.transform = transform

    def __len__(self):
        return len(self.stimulus_names)

    def __getitem__(self, index: int):
        stimulus_name = self.stimulus_names[index]
        decoded_features = {}
        for layer_path_name in self.layer_path_names:
            decoded_feature = self.decoded_features_store.get(
                layer=layer_path_name,
                label=stimulus_name,
                subject=self.subject_id,
                roi=self.roi,
            )
            decoded_feature = decoded_feature[0]  # NOTE: remove batch axis
            decoded_features[layer_path_name] = decoded_feature
        if self.transform is not None:
            decoded_features = self.transform(decoded_features)
        return decoded_features


class ImageDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        stimulus_names: list[str] | None = None,
        extension: str = ".jpg",
    ):
        self.root_path = root_path
        if stimulus_names is None:
            stimulus_names = [
                path.name.removesuffix(extension)
                for path in Path(root_path).glob(f"*{extension}")
            ]
        self.stimulus_names = stimulus_names
        self.extension = extension

    def __len__(self):
        return len(self.stimulus_names)

    def __getitem__(self, index: int):
        stimulus_name = self.stimulus_names[index]
        image = Image.open(Path(self.root_path) / f"{stimulus_name}{self.extension}")
        image = image.convert("RGB")
        return np.array(image) / 255.0, stimulus_name


class RenameFeatureKeys:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def __call__(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {self.mapping.get(key, key): value for key, value in features.items()}
