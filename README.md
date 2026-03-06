# Scalable YOLOv8 Inference with FastAPI & NVIDIA Triton Server

This project is a production-ready system capable of serving an object detection model (YOLOv8) via NVIDIA Triton Inference Server using a high-concurrency FastAPI frontend.

## 🚀 Key Features
- **YOLOv8 Object Detection**: Leverages ONNX export of YOLOv8 for seamless edge-to-server inference capabilities.
- **NVIDIA Triton Inference Server**: Centralized model serving using gRPC communication for lowest latency.
- **Dynamic Batching**: Fully configured Triton `config.pbtxt` to dynamically batch requests during heavy load periods (improves GPU/CPU utilization).
- **Concurrency & Scalability**: FastAPI runs asynchronously with Uvicorn utilizing multiple workers, routing multiple concurrent image inference requests down to Triton.
- **Robustness**: 
  - Comprehensive health-check endpoint (`/health`) returning operational status.
  - Graceful degradation: If Triton goes offline, the API appropriately handles failures and returns structured responses.
- **Prometheus Metrics**: In addition to Triton's native metrics, FastAPI instances export Prometheus metrics at `/metrics`.

---

## 🛠️ Hardware Requirements
- **OS**: Linux / macOS
- **CPU/RAM**: Any standard CPU (Quad-core) + 8GB RAM minimum.
- **GPU (Optional but recommended)**: NVIDIA GPU with latest drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). (If you wish to use GPU, simply uncomment the GPU device bounds in `docker-compose.yml`!).

## 📦 Setup & Execution Steps

### 1. Start the Environment
Run the Docker Compose command. The system automatically provisions an initialization container (`model-exporter`) to download YOLOv8 and compile it to ONNX, configuring the Triton `model_repository`.

```bash
docker-compose up -d --build
```

### 2. Verify Services
Check if the services are healthy. Note that Triton takes a few seconds to load the AI model.

Health Check:
```bash
curl http://localhost:8080/health
```
*(Should return `{"status":"ok","triton":"ready"}` once running).*

### 3. Test Inference
Send a sample image for inference.

1. Download a sample image:
```bash
curl -L -o sample.jpg "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
```
2. Request prediction:
```bash
curl -X POST -F "image=@sample.jpg" http://localhost:8080/predict
```
*(Returns structured JSON detecting people/objects with latency metrics!)*

---

## ⚡ Load Testing (10+ Concurrent Requests)

We use **K6** to perform robust load testing. 

1. Install [k6](https://k6.io/docs/get-started/installation/).
2. Run the provided load test script:
```bash
cd load_test
# Note: ensure 'sample.jpg' is in the load_test directory. Run: curl -L -o sample.jpg "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
k6 run k6_test.js
```

### Expected Output Logs
❯ k6 run k6_test.js

         /\      Grafana   /‾‾/  
    /\  /  \     |\  __   /  /   
   /  \/    \    | |/ /  /   ‾‾\ 
  /          \   |   (  |  (‾)  |
 / __________ \  |_|\_\  \_____/ 


     execution: local
        script: k6_test.js
        output: -

     scenarios: (100.00%) 1 scenario, 15 max VUs, 1m0s max duration (incl. graceful stop):
              * default: 15 looping VUs for 30s (gracefulStop: 30s)



  █ THRESHOLDS 

    http_req_duration
    ✓ 'p(95)<3000' p(95)=2.39s


  █ TOTAL RESULTS 

    checks_total.......: 717     22.591598/s
    checks_succeeded...: 100.00% 717 out of 717
    checks_failed......: 0.00%   0 out of 717

    ✓ is status 200
    ✓ is success true
    ✓ has latency info

    HTTP
    http_req_duration..............: avg=969.75ms min=89.44ms med=919.47ms max=2.69s p(90)=1.62s p(95)=2.39s
      { expected_response:true }...: avg=969.75ms min=89.44ms med=919.47ms max=2.69s p(90)=1.62s p(95)=2.39s
    http_req_failed................: 0.00%  0 out of 239
    http_reqs......................: 239    7.530533/s

    EXECUTION
    iteration_duration.............: avg=1.97s    min=1.09s   med=1.92s    max=3.69s p(90)=2.63s p(95)=3.39s
    iterations.....................: 239    7.530533/s
    vus............................: 11     min=11       max=15
    vus_max........................: 15     min=15       max=15

    NETWORK
    data_received..................: 183 kB 5.8 kB/s
    data_sent......................: 41 MB  1.3 MB/s




running (0m31.7s), 00/15 VUs, 239 complete and 0 interrupted iterations
default ✓ [======================================] 15 VUs  30s
This confirms robust concurrent operations successfully matching the ≥10 request limit handling. 

---

## 🏗️ Horizontal Scaling Architecture Explanation

To scale this application in a production Kubernetes setup for massive traffic, we segment scaling into two vectors:

1. **Frontend FastAPI Instances (Stateless CPU bounds)** 
   - Deploy as a Deployment object with a generic CPU-based **Horizontal Pod Autoscaler (HPA)**.
   - When requests spike, K8s spins up more FastAPI pod replicas handling I/O preprocessing & parsing.
   - A standard Kubernetes ingress or Load Balancer directs HTTP traffic cleanly across them.

2. **Backend Triton Inference Server Instances (Stateful GPU bounds)**
   - Deployed independently from FastAPI. They expose gRPC cluster IPs.
   - Auto-scaled on custom metrics (e.g., *Dynamic Batch Queue Size* from Prometheus or GPU Utilization).
   - If Dynamic Batching hits max thresholds, new Triton servers spin up to consume the batch limits securely.

By separating the heavy Python I/O blocking API operations from the highly optimized C++ Triton GPU calculations into separate container microservices, we dramatically reduce costs by avoiding scaling expensive GPU nodes just to run pre-processing code!

---

## 🔧 Optional Advanced Capabilities Included
* **Dynamic Batching:** Implemented in `scripts/export_yolo.py`. The config builds queue windows caching requests so GPU inference pushes batches at a time, preventing request starvation.
* **Health Endpoint:** Added `/health` evaluating Triton connectivity, allowing Kubernetes deployments to evict degraded pods quickly.
* **Graceful Degradation:** `/predict` handles backend failures with HTTP exceptions if Triton drops load, preventing zombie connections.
* **Metrics:** Endpoints expose Prometheus metrics mapping Triton model latency queues.
