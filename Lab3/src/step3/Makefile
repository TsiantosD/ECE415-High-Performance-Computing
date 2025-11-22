# ==========================================
# CUDA Makefile for Convolution2D
# ==========================================

# Compiler settings
NVCC := nvcc

# Target executable name
TARGET := convolution2d

# Source file
SRC := Convolution2D.cu

# ------------------------------------------
# Flags Configuration
# ------------------------------------------

# Standard Compilation Flags
# -fmad=false This prevents fp precision loss
COMMON_FLAGS := -O4 -fmad=false 

# Debugging Flags
DEBUG_FLAGS := -g -G

# ------------------------------------------
# Build Rules
# ------------------------------------------

# 1. Default Build (Optimized, No Debug)
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(COMMON_FLAGS) -o $(TARGET) $(SRC)

# 2. Debug Build
# Usage: make debug
debug: COMMON_FLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean Build Files
# Usage: make clean
clean:
	rm -f $(TARGET)
