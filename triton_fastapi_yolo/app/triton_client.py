import tritonclient.grpc.aio as grpcclient
import numpy as np
import time

class TritonClient:
    def __init__(self, url="localhost:8001", model_name="yolov8_onnx"):
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=self.url)

    async def infer(self, input_tensor: np.ndarray):
        start_time = time.time()
        
        # Prepare inputs
        inputs = [
            grpcclient.InferInput("images", input_tensor.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_tensor)
        
        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("output0")
        ]
        
        # Inference
        response = await self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        result_tensor = response.as_numpy("output0")
        latency = time.time() - start_time
        
        return result_tensor, latency

    async def is_server_ready(self):
        try:
            return await self.client.is_server_ready()
        except Exception as e:
            return False
