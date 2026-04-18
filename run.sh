#!/bin/bash

echo "Cleaning old builds..."
make clean

echo "Compiling CUDA project..."
make

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running pipeline..."
    echo "-------------------------------------------"
    ./yolo_pipeline
    echo "-------------------------------------------"
else
    echo "Compilation failed."
fi