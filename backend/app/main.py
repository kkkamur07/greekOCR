from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import io

from .models import (
    SegmentRequest, SegmentResponse, Region,
    TranscribeRequest, TranscribeResponse,
    UploadResponse
)
from .services import ImageService, SegmentationService, TranscriptionService

# Create FastAPI app
app = FastAPI(
    title="Greek OCR API",
    description="API for Greek manuscript OCR and transcription",
    version="1.0.0"
)

# CORS - Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Services
image_service = ImageService()
segmentation_service = SegmentationService()
transcription_service = TranscriptionService()


@app.get("/")
def root():
    """Health check"""
    return {"status": "ok", "message": "Greek OCR API is running"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for processing"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Read and save image
        content = await file.read()
        image = image_service.save_image(io.BytesIO(content), image_id, UPLOAD_DIR)
        
        return UploadResponse(
            image_id=image_id,
            width=image.width,
            height=image.height,
            message="Image uploaded successfully"
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error uploading image: {str(e)}")


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Get uploaded image by ID"""
    
    image_path = UPLOAD_DIR / f"{image_id}.png"
    
    if not image_path.exists():
        raise HTTPException(404, "Image not found")
    
    return FileResponse(image_path)


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest):
    """Segment image into text regions"""
    
    try:
        # Load image
        image = image_service.load_image(request.image_id, UPLOAD_DIR)
        
        # Segment
        regions = segmentation_service.segment(
            image,
            device='cpu',  # TODO: Make configurable
            min_area=request.min_area,
            min_width=request.min_width,
            min_height=request.min_height
        )
        
        # Convert to Pydantic models
        region_models = [Region(**r) for r in regions]
        
        return SegmentResponse(
            image_id=request.image_id,
            regions=region_models,
            total_regions=len(region_models)
        )
    
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    except Exception as e:
        raise HTTPException(500, f"Error during segmentation: {str(e)}")


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """Transcribe text regions"""
    
    try:
        # Load image (for future real transcription)
        image = image_service.load_image(request.image_id, UPLOAD_DIR)
        
        # Convert Pydantic models to dicts
        regions = [r.model_dump() for r in request.regions]
        
        # Transcribe (mock for now)
        transcriptions = transcription_service.transcribe_mock(regions)
        
        return TranscribeResponse(
            image_id=request.image_id,
            transcriptions=transcriptions
        )
    
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    except Exception as e:
        raise HTTPException(500, f"Error during transcription: {str(e)}")



@app.post("/api/binarize")
async def binarize_image(request: dict):
    """Binarize image using Kraken"""
    from kraken import binarization
    
    try:
        image_id = request.get('image_id')
        
        # Load image
        image = image_service.load_image(image_id, UPLOAD_DIR)
        
        # Binarize using Kraken
        binarized = binarization.nlbin(image)
        
        # Save binarized version
        binarized_id = f"{image_id}_binarized"
        binarized_path = UPLOAD_DIR / f"{binarized_id}.png"
        binarized.save(binarized_path)
        
        return {
            "image_id": binarized_id,
            "original_id": image_id,
            "message": "Image binarized successfully"
        }
    
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    except Exception as e:
        raise HTTPException(500, f"Error during binarization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)