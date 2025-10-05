# Makefile for simple PTX kernel example

# CUDA compiler
NVCC = nvcc

# Compiler flags
# -arch=sm_80 for Ampere (A100, RTX 30xx)
# -arch=sm_86 for RTX 30xx
# -arch=sm_89 for RTX 40xx
# Adjust based on your GPU
NVCCFLAGS = -arch=sm_80 -O3

# Target executable
TARGET = simple_mma

# Source file
SRC = simple.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

