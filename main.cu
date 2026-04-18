#include "yolo_pipeline.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <sys/stat.h>

// Define STB_IMAGE implementation exactly once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ---------------------------------------------------------
// CUSTOM CUDA KERNEL
// Converts Interleaved 8-bit BGR (HWC) to Planar 32-bit Float RGB (CHW)
// and normalizes pixels from [0, 255] to [0.0, 1.0]
// ---------------------------------------------------------
__global__ void preprocessYoloKernel(
    const unsigned char* d_input_HWC, 
    float* d_output_CHW, 
    int width, 
    int height,
    int batch_size) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x < width && y < height && b < batch_size) {
        int area = width * height;
        
        int img_offset_in = b * area * 3;
        int img_offset_out = b * area * 3;

        int in_idx = img_offset_in + (y * width + x) * 3;
        
        int out_idx_R = img_offset_out + 0 * area + (y * width + x);
        int out_idx_G = img_offset_out + 1 * area + (y * width + x);
        int out_idx_B = img_offset_out + 2 * area + (y * width + x);

        // STB Image loads as RGB by default (unlike OpenCV which is BGR)
        // So we read RGB and write RGB directly.
        d_output_CHW[out_idx_R] = d_input_HWC[in_idx + 0] / 255.0f; // R
        d_output_CHW[out_idx_G] = d_input_HWC[in_idx + 1] / 255.0f; // G
        d_output_CHW[out_idx_B] = d_input_HWC[in_idx + 2] / 255.0f; // B
    }
}

// ---------------------------------------------------------
// HOST PIPELINE FUNCTION
// ---------------------------------------------------------
void runYoloPreProcessingPipeline() {
    std::cout << "Scanning 'data/' folder for images..." << std::endl;
    std::vector<std::string> image_files;
    
    // Read the 'data' directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir("data")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fname = ent->d_name;
            if (fname.find(".jpg") != std::string::npos || fname.find(".jpeg") != std::string::npos) {
                image_files.push_back("data/" + fname);
                if (image_files.size() >= TOTAL_IMAGES) break; // Cap at TOTAL_IMAGES
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open directory 'data'. Make sure it exists." << std::endl;
        exit(EXIT_FAILURE);
    }

    int actual_images = image_files.size();
    if (actual_images == 0) {
        std::cerr << "Error: No .jpg files found in the 'data/' folder." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Round down to the nearest full batch to prevent memory access violations
    int num_batches = actual_images / BATCH_SIZE;
    int processed_images = num_batches * BATCH_SIZE;

    if (num_batches == 0) {
        std::cerr << "Error: Not enough images to fill a single batch of " << BATCH_SIZE << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t bytes_per_image_in = YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS * sizeof(unsigned char);
    size_t bytes_per_image_out = YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS * sizeof(float);
    
    size_t batch_bytes_in = BATCH_SIZE * bytes_per_image_in;
    size_t batch_bytes_out = BATCH_SIZE * bytes_per_image_out;

    std::cout << "Found " << actual_images << " images. Processing " << processed_images 
              << " images in " << num_batches << " batches." << std::endl;

    // 1. Allocate Pinned Host Memory
    unsigned char* h_input;
    float* h_output;
    CUDA_CHECK(cudaMallocHost((void**)&h_input, processed_images * bytes_per_image_in));
    CUDA_CHECK(cudaMallocHost((void**)&h_output, processed_images * bytes_per_image_out));

    // Load and Resize images into Host Memory
    std::cout << "Loading and resizing images to 640x640 on CPU..." << std::endl;
    for (int i = 0; i < processed_images; ++i) {
        int width, height, channels;
        unsigned char* img_data = stbi_load(image_files[i].c_str(), &width, &height, &channels, YOLO_CHANNELS);
        
        if (img_data == nullptr) {
            std::cerr << "Failed to load: " << image_files[i] << std::endl;
            continue;
        }

        // Basic nearest-neighbor resize to force image into 640x640 constraint
        size_t offset = i * YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS;
        float x_ratio = (float)width / YOLO_WIDTH;
        float y_ratio = (float)height / YOLO_HEIGHT;
        
        for (int y = 0; y < YOLO_HEIGHT; y++) {
            for (int x = 0; x < YOLO_WIDTH; x++) {
                int px = std::min((int)(x * x_ratio), width - 1);
                int py = std::min((int)(y * y_ratio), height - 1);
                
                for (int c = 0; c < YOLO_CHANNELS; c++) {
                    h_input[offset + (y * YOLO_WIDTH + x) * YOLO_CHANNELS + c] = 
                        img_data[(py * width + px) * YOLO_CHANNELS + c];
                }
            }
        }
        stbi_image_free(img_data);
    }

    // 2. Allocate Device Memory (For all streams)
    unsigned char* d_input[NUM_STREAMS];
    float* d_output[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&d_input[i], batch_bytes_in));
        CUDA_CHECK(cudaMalloc((void**)&d_output[i], batch_bytes_out));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((YOLO_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (YOLO_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   BATCH_SIZE);

    // 3. Execute Pipeline with Overlapping Streams
    std::cout << "Starting Multi-Stream GPU processing..." << std::endl;
    for (int i = 0; i < num_batches; ++i) {
        int stream_idx = i % NUM_STREAMS;
        int offset_in = i * BATCH_SIZE * YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS;
        int offset_out = offset_in; 

        CUDA_CHECK(cudaMemcpyAsync(d_input[stream_idx], &h_input[offset_in], batch_bytes_in, cudaMemcpyHostToDevice, streams[stream_idx]));

        preprocessYoloKernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            d_input[stream_idx], 
            d_output[stream_idx], 
            YOLO_WIDTH, 
            YOLO_HEIGHT, 
            BATCH_SIZE
        );

        CUDA_CHECK(cudaMemcpyAsync(&h_output[offset_out], d_output[stream_idx], batch_bytes_out, cudaMemcpyDeviceToHost, streams[stream_idx]));
        
        std::cout << "Dispatched Batch " << i + 1 << "/" << num_batches << " to Stream " << stream_idx << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "All batches processed successfully!" << std::endl;

    // --- ADD THIS BLOCK ---
    std::cout << "Saving processed float tensors to disk..." << std::endl;
    mkdir("data/output", 0777);
    for (int i = 0; i < processed_images; ++i) {
        char out_filename[256];
        // Name them tensor_0000.bin, tensor_0001.bin, etc.
        snprintf(out_filename, sizeof(out_filename), "data/output/tensor_%04d.bin", i);
        
        FILE* f = fopen(out_filename, "wb");
        if (f) {
            size_t offset = i * YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS;
            // Write the raw 32-bit floats directly to disk
            fwrite(&h_output[offset], sizeof(float), YOLO_WIDTH * YOLO_HEIGHT * YOLO_CHANNELS, f);
            fclose(f);
        } else {
            std::cerr << "Could not open file for writing: " << out_filename << std::endl;
        }
    }
    std::cout << "Successfully saved all tensors to data/output/!" << std::endl;
    // ----------------------

    // 4. Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_input[i]));
        CUDA_CHECK(cudaFree(d_output[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
}

int main() {
    runYoloPreProcessingPipeline();
    return 0;
}