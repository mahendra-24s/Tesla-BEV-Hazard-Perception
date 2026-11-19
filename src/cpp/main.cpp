#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

struct Detection3D {
    int class_id;
    float confidence;
    float x_forward_m;
    float y_lateral_m;
};

// Back-projects bottom-center pixel (u, v) onto ego-vehicle ground plane (Z = 0)
bool projectPixelToBEV(float u, float v, const cv::Matx33f& K_inv,
                       const cv::Matx33f& R_cam_to_ego, const cv::Vec3f& t_cam_to_ego,
                       float& out_x_fwd, float& out_y_lat) {
    cv::Vec3f uv_hom(u, v, 1.0f);
    cv::Vec3f ray_cam = K_inv * uv_hom;
    cv::Vec3f ray_ego = R_cam_to_ego * ray_cam;

    if (ray_ego[2] >= -1e-3f) {
        return false; // Ray points above horizon
    }

    float scale = -t_cam_to_ego[2] / ray_ego[2];
    cv::Vec3f pt_ego = t_cam_to_ego + scale * ray_ego;

    if (std::abs(pt_ego[0]) > 55.0f || std::abs(pt_ego[1]) > 55.0f) {
        return false;
    }

    out_x_fwd = pt_ego[0];
    out_y_lat = pt_ego[1];
    return true;
}

// Normalizes BGR OpenCV image into CHW float32 tensor with ImageNet mean/std
void preprocessImage(const cv::Mat& bgr, std::vector<float>& host_buffer, int target_size = 560) {
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(target_size, target_size));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};
    int area = target_size * target_size;

    const float* rgb_ptr = rgb.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < area; ++i) {
            float val = rgb_ptr[i * 3 + c];
            host_buffer[c * area + i] = (val - mean[c]) / std_val[c];
        }
    }
}

int main(int argc, char** argv) {
    std::string enginePath = (argc > 1) ? argv[1] : "../models/engines/rfdetr_rtx5050.engine";

    // 1. Load serialized TensorRT engine
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "[ERROR] Cannot open TensorRT engine: " << enginePath << std::endl;
        return 1;
    }
    std::streamsize size = file.tellg();
    if (size <= 0) {
        std::cerr << "[ERROR] Empty or unreadable TensorRT engine: " << enginePath << std::endl;
        return 1;
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(size));
    file.read(engineData.data(), size);

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime ? runtime->deserializeCudaEngine(engineData.data(), static_cast<size_t>(size)) : nullptr;
    IExecutionContext* context = engine ? engine->createExecutionContext() : nullptr;

    if (!context) {
        std::cerr << "[ERROR] Failed to deserialize TensorRT engine or create execution context." << std::endl;
        delete engine;
        delete runtime;
        return 1;
    }

    // 2. Allocate host and device buffers matching RF-DETR tensor signatures
    const int kInputElems = 1 * 3 * 560 * 560;
    const int kDetsElems  = 1 * 300 * 4;
    const int kLabelElems = 1 * 300 * 30;

    std::vector<float> h_input(kInputElems, 0.0f);
    std::vector<float> h_dets(kDetsElems, 0.0f);
    std::vector<float> h_labels(kLabelElems, 0.0f);

    void *d_input = nullptr, *d_dets = nullptr, *d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  kInputElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dets,   kDetsElems  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, kLabelElems * sizeof(float)));

    if (!context->setTensorAddress("input", d_input) ||
        !context->setTensorAddress("dets", d_dets) ||
        !context->setTensorAddress("labels", d_labels)) {
        std::cerr << "[ERROR] Failed to bind one or more TensorRT tensor addresses." << std::endl;
        cudaFree(d_input);
        cudaFree(d_dets);
        cudaFree(d_labels);
        delete context;
        delete engine;
        delete runtime;
        return 1;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 3. Optional real image inference if image path is passed as argv[2]
    if (argc > 2) {
        cv::Mat frame = cv::imread(argv[2]);
        if (frame.empty()) {
            std::cerr << "[WARN] Failed to load image: " << argv[2] << std::endl;
        } else {
            preprocessImage(frame, h_input, 560);
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), kInputElems * sizeof(float), cudaMemcpyHostToDevice, stream));
            if (!context->enqueueV3(stream)) {
                std::cerr << "[ERROR] TensorRT enqueueV3 failed during image inference." << std::endl;
            }
            CUDA_CHECK(cudaMemcpyAsync(h_dets.data(), d_dets, kDetsElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_labels.data(), d_labels, kLabelElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Example front-camera calibration (nuScenes CAM_FRONT nominal)
            cv::Matx33f K(1266.4f, 0.0f, 816.2f, 0.0f, 1266.4f, 491.5f, 0.0f, 0.0f, 1.0f);
            cv::Matx33f R(0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f);
            cv::Vec3f t(1.70f, 0.0f, 1.51f);
            cv::Matx33f K_inv = K.inv();

            std::vector<Detection3D> detections;
            detections.reserve(64);

            for (int q = 0; q < 300; ++q) {
                float best_prob = 0.0f;
                int best_cls = 0;
                for (int c = 0; c < 30; ++c) {
                    float prob = 1.0f / (1.0f + std::exp(-h_labels[q * 30 + c]));
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_cls = c;
                    }
                }
                if (best_prob > 0.40f) {
                    float cx = h_dets[q * 4 + 0] * static_cast<float>(frame.cols);
                    float cy = h_dets[q * 4 + 1] * static_cast<float>(frame.rows);
                    float h  = h_dets[q * 4 + 3] * static_cast<float>(frame.rows);
                    float x_fwd = 0.0f, y_lat = 0.0f;

                    if (projectPixelToBEV(cx, cy + h * 0.5f, K_inv, R, t, x_fwd, y_lat)) {
                        detections.push_back({best_cls, best_prob, x_fwd, y_lat});
                    }
                }
            }

            for (const auto& det : detections) {
                std::cout << "[3D HAZARD] Class " << det.class_id
                          << " (" << det.confidence * 100.0f << "%) -> BEV X="
                          << det.x_forward_m << "m, Y=" << det.y_lateral_m << "m\n";
            }
            std::cout << "[INFO] Projected " << detections.size() << " hazards into 3D BEV space.\n";
        }
    }

    // 4. 100-iteration pure GPU latency benchmark
    for (int i = 0; i < 20; ++i) {
        context->enqueueV3(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        context->enqueueV3(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double mean_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    std::cout << "[BENCHMARK] C++17 TensorRT Mean Latency: " << mean_ms
              << " ms (" << (1000.0 / mean_ms) << " FPS)" << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_dets));
    CUDA_CHECK(cudaFree(d_labels));
    delete context;
    delete engine;
    delete runtime;
    return 0;
}