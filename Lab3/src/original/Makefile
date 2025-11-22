# ==========================================
# CUDA Makefile for Convolution2D
# ==========================================

# Compiler settings
NVCC := gcc

# Target executable name
TARGET := convolution2d

# Source file
SRC := Convolution2D.c

# ------------------------------------------
# Flags Configuration
# ------------------------------------------

# Standard Compilation Flags
COMMON_FLAGS := -O4

# Debugging Flags
DEBUG_FLAGS := -g -G

# ------------------------------------------
# Build Rules
# ------------------------------------------

# 1. Default Build (Optimized, No Debug)
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -Wall $(COMMON_FLAGS) -o $(TARGET) $(SRC)

# 2. Debug Build
# Usage: make debug
debug: COMMON_FLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean Build Files
# Usage: make clean
clean:
	rm -f $(TARGET)