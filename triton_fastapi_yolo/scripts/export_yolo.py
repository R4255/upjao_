import os
import shutil
from ultralytics import YOLO

def export_model():
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # Export to ONNX with dynamic axes for batch size
    print("Exporting to ONNX...")
    # By default, ultralytics export dynamic batch if dynamic=True.
    onnx_file = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"Exported to {onnx_file}")
    
    # Move to Triton Model Repository
    model_repo = "/model_repository/yolov8_onnx"
    version_dir = os.path.join(model_repo, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    target_onnx = os.path.join(version_dir, "model.onnx")
    print(f"Moving {onnx_file} to {target_onnx}")
    shutil.move(onnx_file, target_onnx)

    # Generate config.pbtxt
    config_str = """name: "yolov8_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]
"""
    config_path = os.path.join(model_repo, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_str)
    
    print("Config written successfully. Triton model repository is ready.")

if __name__ == "__main__":
    export_model()
