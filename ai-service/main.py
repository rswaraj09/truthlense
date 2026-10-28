import os
import io
import time
import uuid
import tempfile
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import model_manager, DEVICE

app = FastAPI(
    title="TruthLens AI Inference Service",
    description="Microservice providing PyTorch model inference for AI-generated and manipulated media detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_DIMENSION = 8192
MAX_VIDEO_FRAMES_SAMPLE = 15

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "TruthLens AI Service",
        "device": str(DEVICE),
        "cuda_available": DEVICE.type == "cuda",
        "model_loaded": model_manager.is_loaded,
        "model_metadata": model_manager.model_metadata
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided file is not a valid image MIME type"
        )

    # 2. Read content safely with size limit
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file size exceeds maximum limit of 25MB"
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty"
        )

    # 3. Open image with Pillow to verify integrity and dimensions
    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.verify()
        # Re-open after verify() because verify() clears file pointer
        pil_image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted image file or unsupported image structure"
        )

    w, h = pil_image.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions ({w}x{h}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION}px"
        )

    # 4. Perform PyTorch inference
    try:
        result = model_manager.predict_pil_image(pil_image)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred during deep learning model inference"
        )

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    start_time = time.time()

    # 1. Validate MIME type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided file is not a valid video MIME type"
        )

    # 2. Stream to temporary file on disk with size protection
    temp_video_path = os.path.join(tempfile.gettempdir(), f"tl_vid_{uuid.uuid4().hex}.mp4")
    total_bytes = 0

    try:
        with open(temp_video_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Video file size exceeds maximum limit of 100MB"
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty"
            )

        # 3. Open video with OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode video. Codec or container may be corrupted."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / fps if total_frames > 0 else 0

        if total_frames <= 0:
            cap.release()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video has no readable frames"
            )

        # 4. Sample frames uniformly across video duration
        sample_count = min(MAX_VIDEO_FRAMES_SAMPLE, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frame_results = []
        scores = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            timestamp_sec = round(idx / fps, 2)
            mins = int(timestamp_sec // 60)
            secs = int(timestamp_sec % 60)
            formatted_time = f"{mins:02d}:{secs:02d}"

            # Convert BGR (OpenCV) to RGB (Pillow)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            inference = model_manager.predict_pil_image(pil_frame)
            score = inference["score"]
            scores.append(score)

            is_flagged = score >= 0.65
            frame_results.append({
                "frameIndex": int(idx),
                "timestampSec": timestamp_sec,
                "timestampFormatted": formatted_time,
                "score": score,
                "prediction": inference["prediction"],
                "flagged": is_flagged
            })

        cap.release()

        if not scores:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract readable frames from video"
            )

        # 5. Aggregate video assessment
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        flagged_count = sum(1 for f in frame_results if f["flagged"])
        flagged_percentage = round((flagged_count / len(frame_results)) * 100, 1)

        # Conservative classification for video preliminary screening
        if max_score >= 0.85 and flagged_percentage >= 25:
            overall_prediction = "ai_generated"
            overall_label = "Likely AI-generated (Multiple Frames Flagged)"
            confidence = "medium"
        elif avg_score >= 0.65:
            overall_prediction = "ai_generated"
            overall_label = "Potentially manipulated / AI-synthesized"
            confidence = "medium"
        elif avg_score <= 0.35 and max_score < 0.60:
            overall_prediction = "authentic"
            overall_label = "Likely authentic"
            confidence = "medium"
        else:
            overall_prediction = "inconclusive"
            overall_label = "Inconclusive / Mixed indicators"
            confidence = "low"

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "prediction": overall_prediction,
            "score": round(avg_score, 4),
            "label": overall_label,
            "confidenceLevel": confidence,
            "processingTimeMs": processing_time_ms,
            "isPreliminaryScreening": True,
            "frames": frame_results,
            "forensicMetrics": {
                "totalFrames": total_frames,
                "analyzedFrames": len(frame_results),
                "fps": round(fps, 2),
                "durationSec": round(duration_sec, 2),
                "flaggedFramesCount": flagged_count,
                "flaggedPercentage": flagged_percentage,
                "peakFakeScore": round(max_score, 4),
                "modelWeightsActive": model_manager.is_loaded
            },
            "limitation": (
                "PRELIMINARY FRAME SCREENING ONLY: This analysis samples individual frames using static image models. "
                "It does not evaluate temporal frame consistency, lip-sync coherence, or audio synthesis. "
                "A dedicated video deepfake architecture should be consulted for binding forensic determinations."
            )
        }

    finally:
        # Mandatory cleanup of temporary video files in finally block
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
