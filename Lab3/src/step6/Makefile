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

# Set FMAD to true if not defined on the command line
FMAD ?= true

# Use of doubles or floats
USE_DOUBLES ?= 1

# Standard Compilation Flags
# Note: Use FMAD=false to prevent fp precision loss if needed
COMMON_FLAGS := -O4 -fmad=$(FMAD) -DUSE_DOUBLES=$(USE_DOUBLES)

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