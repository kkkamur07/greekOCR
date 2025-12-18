import torch
from PIL import Image
import numpy as np
from kraken.blla import segment
from kraken.lib.vgsl import TorchVGSLModel
from importlib import resources


def load_segmentation_model(device='cpu'):
    model_path = resources.files('kraken').joinpath('blla.mlmodel')
    seg_model = TorchVGSLModel.load_model(str(model_path))
    return seg_model


def segment_image(image, device='cpu'):

    seg_model = load_segmentation_model(device)
    segmented = segment(im=image, device=device, model=seg_model)
    return segmented


def extract_data(segmented_image):

    lines_data = []
    
    for idx, line in enumerate(segmented_image.lines):
        boundary = np.array(line.boundary)
        x_min = int(boundary[:, 0].min())
        y_min = int(boundary[:, 1].min())
        x_max = int(boundary[:, 0].max())
        y_max = int(boundary[:, 1].max())
        
        lines_data.append({
            'id': idx,
            'baseline': line.baseline,
            'boundary': line.boundary,
            'bbox': (x_min, y_min, x_max, y_max),
            'tags': line.tags
        })
    
    regions_data = []
    for region in segmented_image.regions:
        if hasattr(region, 'boundary') and region.boundary:
            regions_data.append({
                'boundary': region.boundary,
                'type': getattr(region, 'type', 'text')
            })
    
    return {
        'lines': lines_data,
        'regions': regions_data,
        'num_lines': len(lines_data),
        'num_regions': len(regions_data)
    }