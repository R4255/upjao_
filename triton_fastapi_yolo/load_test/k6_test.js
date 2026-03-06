import http from 'k6/http';
import { check, sleep } from 'k6';

// Read an image file beforehand to send in each request
// Make sure this file exists in the directory where you run k6
const img = open('./sample.jpg', 'b');

export let options = {
    // Handling at minimum 10 concurrent requests as requested
    vus: 15,
    duration: '30s',
    thresholds: {
        // 95% of requests should complete within 3 seconds
        // (Since Triton latency depends heavily on hardware, we are setting this high for CPU fallback).
        http_req_duration: ['p(95)<3000'],
    },
};

export default function () {
    const url = 'http://localhost:8080/predict';
    
    // Create multipart form payload
    const payload = {
        image: http.file(img, 'sample.jpg', 'image/jpeg')
    };

    const res = http.post(url, payload);

    // Verify successful responses
    check(res, {
        'is status 200': (r) => r.status === 200,
        'is success true': (r) => {
             const body = JSON.parse(r.body);
             return body.success === true;
        },
        'has latency info': (r) => {
             const body = JSON.parse(r.body);
             return body.latency !== undefined;
        }
    });

    sleep(1);
}
