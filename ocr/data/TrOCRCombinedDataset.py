import bisect
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Iterable, List, Tuple, Union
from pathlib import Path

class TrOCRCombinedDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset], processor, max_target_length: int = 128):

        self.datasets: List[Dataset] = list(datasets)
        self.processor = processor
        self.max_target_length = max_target_length

        self.cumulative_sizes = []
        total = 0
        
        for d in self.datasets:
            total += len(d)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _map_index(self, idx: int) -> Tuple[int, int]:

        if idx < 0:
            idx = len(self) + idx
        
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        return dataset_idx, local_idx

    def __getitem__(self, idx: int):
        dataset_idx, local_idx = self._map_index(idx)
        
        image_obj, text = self.datasets[dataset_idx][local_idx]

        if isinstance(image_obj, (str, Path)):
            image = Image.open(image_obj).convert("RGB")
        elif isinstance(image_obj, Image.Image):
            image = image_obj.convert("RGB")
        else:
            raise TypeError(f"Expected path or PIL Image, got {type(image_obj)}")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }
