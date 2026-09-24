import os
import torch
import traceback
from datetime import datetime
import uuid

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from diffusers import DiffusionPipeline
from huggingface_hub import login
from moviepy import ImageSequenceClip

from db import videos_collection  # MongoDB collection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")  # load from environment, not hardcoded
pipe = None
OUTPUT_DIR = "video_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_filename():
    return os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex[:8]}_video.mp4")

def load_pipeline():
    global pipe
    print("[INFO] Loading video generation model...")
    try:
        login(HF_TOKEN)
        pipe = DiffusionPipeline.from_pretrained(
            "strangeman3107/animov-512x",  # anime/cartoon-style text-to-video model
            use_auth_token=True,
            force_download=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print("[SUCCESS] Model loaded successfully.")
    except Exception:
        print("[ERROR] Failed to load model:")
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    load_pipeline()

@app.post("/api/cartoon/generate")
async def generate(prompt: str = Form(...), duration: int = Form(5)):
    print(f"[REQUEST] Prompt received: {prompt}, Duration: {duration}s")

    if pipe is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    try:
        fps = 5
        target_frames = duration * fps
        print(f"[INFO] Generating {target_frames} frames...")

        result = pipe(prompt, num_inference_steps=25, num_frames=target_frames)
        frames = result["frames"] if "frames" in result else result["images"]
        frames = frames[:target_frames]  # Clip exactly to requested duration

        filename = generate_filename()
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(filename, codec="libx264", verbose=False, logger=None)

        # Save metadata to MongoDB
        metadata = {
            "prompt": prompt,
            "duration": duration,
            "num_frames": len(frames),
            "fps": fps,
            "file_path": filename.replace("\\", "/"),
            "created_at": datetime.utcnow()
        }
        await videos_collection.insert_one(metadata)

        return FileResponse(filename, media_type="video/mp4", filename=os.path.basename(filename))

    except Exception as e:
        print("[ERROR] Exception during video generation:")
        traceback.print_exc()
        return JSONResponse(content={
            "error": "Failed to generate video",
            "details": str(e)
        }, status_code=500)

@app.get("/api/cartoon/videos")
async def get_videos():
    try:
        raw = await videos_collection.find().to_list(length=100)
        videos = []
        for video in raw:
            video["_id"] = str(video["_id"])
            video["created_at"] = video["created_at"].isoformat()
            videos.append(video)
        return videos
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Cartoon Video Generator is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)
