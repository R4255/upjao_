from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import time
import asyncio
from triton_client import TritonClient
from utils import preprocess_image, postprocess_output
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Inference API via Triton Integration")

triton_url = os.getenv("TRITON_SERVER_URL", "tritonserver:8001")
triton_client = TritonClient(url=triton_url, model_name="yolov8_onnx")

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
async def startup_event():
    # Attempt to wait for Triton briefly but don't block indefinitely
    logger.info("Initializing application and checking Triton status...")
    for _ in range(5):
        if await triton_client.is_server_ready():
            logger.info("Triton inference server is online and ready.")
            return
        await asyncio.sleep(2)
        logger.warning("Waiting for Triton server...")

@app.get("/health")
async def health_check():
    """Health check endpoint. Checks if both FastAPI & Triton are ready."""
    is_ready = await triton_client.is_server_ready()
    if is_ready:
        return {"status": "ok", "triton": "ready"}
    else:
        # Return 200 with degraded state, or 503 depending on requirements
        # Returning 503 is standard for Kubernetes liveness/readiness probes
        return JSONResponse(status_code=503, content={"status": "degraded", "triton": "unavailable"})

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    """Predict objects in the uploaded image using YOLOv8 via Triton."""
    if not image.content_type.startswith("image/"):
        logger.error(f"Invalid content type: {image.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    # Graceful degradation if Triton is down
    if not await triton_client.is_server_ready():
        logger.error("Predict called but Triton server is unavailable.")
        raise HTTPException(status_code=503, detail="Triton inference server is temporarily unavailable.")
    
    start_time = time.time()
    
    # Read uploaded file
    try:
        img_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read image bytes: {e}")
        raise HTTPException(status_code=400, detail="Could not read uploaded image data.")
    
    # Preprocess
    try:
        input_tensor, orig_shape = preprocess_image(img_bytes)
        preprocess_time = time.time() - start_time
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Image preprocessing failed.")
    
    # Inference via Triton
    try:
        output_tensor, infer_latency = await triton_client.infer(input_tensor)
    except Exception as e:
        logger.error(f"Triton inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    # Postprocess
    try:
        t0 = time.time()
        results = postprocess_output(output_tensor, orig_shape)
        postprocess_time = time.time() - t0
    except Exception as e:
        logger.error(f"Postprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Image postprocessing failed.")
    
    total_time = time.time() - start_time
    
    logger.info(f"Successfully processed image. Latency: {round(total_time, 4)}s. Detections: {len(results)}")
    
    return {
        "success": True,
        "detections": results,
        "latency": {
            "total_sec": round(total_time, 4),
            "inference_sec": round(infer_latency, 4),
            "preprocess_sec": round(preprocess_time, 4),
            "postprocess_sec": round(postprocess_time, 4)
        }
    }
