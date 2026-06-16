import axios from 'axios';
import { UploadResponse, SegmentResponse, TranscribeResponse, Region } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadImage = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<UploadResponse>(
    `${API_BASE_URL}/api/upload`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

export const segmentImage = async (
  imageId: string,
  minArea: number = 500,
  minWidth: number = 30,
  minHeight: number = 15
): Promise<SegmentResponse> => {
  const response = await api.post<SegmentResponse>('/api/segment', {
    image_id: imageId,
    min_area: minArea,
    min_width: minWidth,
    min_height: minHeight,
  });

  return response.data;
};

export const transcribeRegions = async (
  imageId: string,
  regions: Region[]
): Promise<TranscribeResponse> => {
  const response = await api.post<TranscribeResponse>('/api/transcribe', {
    image_id: imageId,
    regions: regions,
  });

  return response.data;
};

// New: Transcribe a single region
export const transcribeSingleRegion = async (
  imageId: string,
  region: Region
): Promise<TranscribeResponse> => {
  const response = await api.post<TranscribeResponse>('/api/transcribe', {
    image_id: imageId,
    regions: [region],
  });

  return response.data;
};

export const getImageUrl = (imageId: string): string => {
  return `${API_BASE_URL}/api/images/${imageId}`;
};

export const binarizeImage = async (imageId: string): Promise<{ image_id: string; original_id: string; message: string }> => {
  const response = await api.post('/api/binarize', {
    image_id: imageId,
  });

  return response.data;
};