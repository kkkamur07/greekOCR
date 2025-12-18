from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import TrOCRProcessor


@dataclass
class TrOCRDataCollator:
    processor: TrOCRProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
        labels = [f["labels"] for f in features]
        max_length = max(len(l) for l in labels)
        
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(padded_labels)
        }