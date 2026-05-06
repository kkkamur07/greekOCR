from pydantic import BaseModel
from typing import List, Optional, Tuple

class SegmentRequest(BaseModel):
    image_id: str
    min_area: int = 500
    min_width: int = 30
    min_height: int = 15

class Region(BaseModel):
    id: int
    boundary: List[List[float]]  # [[x, y], [x, y], ...]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

class SegmentResponse(BaseModel):
    image_id: str
    regions: List[Region]
    total_regions: int

class TranscribeRequest(BaseModel):
    image_id: str
    regions: List[Region]

class Transcription(BaseModel):
    region_id: int
    text: str
    confidence: float

class TranscribeResponse(BaseModel):
    image_id: str
    transcriptions: List[Transcription]

class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    message: str