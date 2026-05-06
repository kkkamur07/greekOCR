import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from kraken.blla import segment
from kraken.lib.vgsl import TorchVGSLModel
from importlib import resources


class ImageService:
    """Handle image operations"""
    
    @staticmethod
    def save_image(file_content, image_id: str, upload_dir: Path) -> Image.Image:
        """Save uploaded image and return PIL Image"""
        upload_dir.mkdir(exist_ok=True)
        
        # Save original image
        image_path = upload_dir / f"{image_id}.png"
        image = Image.open(file_content)
        image.save(image_path)
        
        return image
    
    @staticmethod
    def load_image(image_id: str, upload_dir: Path) -> Image.Image:
        """Load image by ID"""
        image_path = upload_dir / f"{image_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_id} not found")
        return Image.open(image_path)


class SegmentationService:
    """Handle segmentation operations"""
    
    _model = None
    
    @classmethod
    def load_model(cls, device='cpu'):
        """Load Kraken segmentation model (cached)"""
        if cls._model is None:
            model_path = resources.files('kraken').joinpath('blla.mlmodel')
            cls._model = TorchVGSLModel.load_model(str(model_path))
        return cls._model
    
    @staticmethod
    def extract_data(segmented_image):
        """Extract region data from segmented image"""
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
        
        return {'lines': lines_data}
    
    @staticmethod
    def filter_regions(lines_data, min_width=20, min_height=10, min_area=200):
        """Filter out regions that are too small"""
        filtered = []
        
        for line in lines_data:
            bbox = line['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            if width >= min_width and height >= min_height and area >= min_area:
                filtered.append(line)
        
        return filtered
    
    @classmethod
    def segment(cls, image: Image.Image, device: str = 'cpu', 
                min_area: int = 500, min_width: int = 30, min_height: int = 15):
        """Segment image and return filtered regions"""
        
        # Load model
        seg_model = cls.load_model(device)
        
        # Run segmentation
        segmented = segment(im=image, device=device, model=seg_model)
        
        # Extract data
        data = cls.extract_data(segmented)
        
        # Apply filters
        filtered_lines = cls.filter_regions(
            data['lines'],
            min_width=min_width,
            min_height=min_height,
            min_area=min_area
        )
        
        # Format regions
        regions = []
        for line in filtered_lines:
            regions.append({
                'id': line['id'],
                'boundary': line['boundary'],
                'bbox': line['bbox']
            })
        
        return regions


class TranscriptionService:
    """Handle OCR transcription"""
    
    @staticmethod
    def transcribe_mock(regions):
        """Mock transcription (replace with real model later)"""
        import random
        
        mock_texts = [
            "Ἐν ἀρχῇ ἦν ὁ λόγος",
            "καὶ ὁ λόγος ἦν πρὸς τὸν θεόν",
            "καὶ θεὸς ἦν ὁ λόγος",
            "οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν",
            "πάντα δι' αὐτοῦ ἐγένετο",
        ]
        
        transcriptions = []
        for region in regions:
            transcriptions.append({
                'region_id': region['id'],
                'text': mock_texts[region['id'] % len(mock_texts)],
                'confidence': random.uniform(0.75, 0.98)
            })
        
        return transcriptions
    
    # TODO: Add real transcription method when model is trained
    # @staticmethod
    # def transcribe_real(image, regions, model, processor, device):
    #     """Real transcription using trained model"""
    #     pass