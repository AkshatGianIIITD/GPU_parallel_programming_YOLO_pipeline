NVCC = nvcc
CFLAGS = -O3 -std=c++14
TARGET = yolo_pipeline
SRCS = main.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)