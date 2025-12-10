# ==========================================
# C/CUDA Makefile for CLAHE Project
# ==========================================

# 1. Compiler settings
CC := nvcc

# 2. Target executable name
TARGET := main

# 3. Source files
# AUTOMATICALLY find all .c files in the current directory
C_SRCS := $(wildcard *.cpp)

# SPECIFY the single .cu file here
# You can change this manually or override it: make CU_FILE=other_kernel.cu
CU_FILE := clahe-3-kernels.cu

# 4. Object files
# Convert all found .c files to .o
C_OBJS := $(C_SRCS:.cpp=.o)
# Convert the specified .cu file to .o
CU_OBJ := $(CU_FILE:.cu=.o)

# Combine them for the linker
OBJS := $(C_OBJS) $(CU_OBJ)

ifeq ($(USER_FLAGS),)
    USER_FLAGS := -DCHECK_OUTPUT=1
endif

# Base Flags
# USER_FLAGS is where the script will inject -DCHECK_OUTPUT
CFLAGS := $(USER_FLAGS)

# ------------------------------------------
# Build Rules
# ------------------------------------------

# --- 1. Default Build (Release) ---
all: CFLAGS += -O3
all: $(TARGET)

# --- 2. Debug Build ---
# -G generates device debug info, -g generates host debug info
debug: CFLAGS += -g -G -O0
debug: $(TARGET)

# --- Link Step ---
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# --- Compile Step for C files ---
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# --- Compile Step for CUDA files ---
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# --- Clean ---
clean:
	rm -f $(TARGET) $(OBJS)