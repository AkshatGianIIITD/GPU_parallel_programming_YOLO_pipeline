#ifndef YOLO_PIPELINE_H
#define YOLO_PIPELINE_H

#include <iostream>
#include <vector>

// YOLO standard input dimensions
const int YOLO_WIDTH = 640;
const int YOLO_HEIGHT = 640;
const int YOLO_CHANNELS = 3;

// Pipeline configuration
const int TOTAL_IMAGES = 100; // Simulating 100s of small/medium images
const int BATCH_SIZE = 10;    // Process 10 images at a time
const int NUM_STREAMS = 3;    // Multi-stream architecture for overlapping I/O and compute

// Function prototype for the main pipeline execution
void runYoloPreProcessingPipeline();

#endif // YOLO_PIPELINE_H