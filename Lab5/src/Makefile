# ================== COMPILERS ==================

CC = gcc
NVCC = nvcc

# ================== FLAGS ==================

# Common optimization flags for CPU (GCC)
COMMON_FLAGS = -Wall -O3 -march=native -ffast-math

# OpenMP flags (For parallel CPU build)
OMP_FLAGS = -fopenmp

# CUDA flags
# -Xcompiler passes flags to the host compiler (gcc). 
# We pass -fopenmp so the host side of the CUDA code can spawn threads.
# --use_fast_math is the CUDA equivalent of -ffast-math
NVCC_FLAGS = -O3 --use_fast_math -Xcompiler -fopenmp

# Math library
LIBS = -lm

# ================== SOURCES ==================

OMP_SRC  = nbody_omp.c
SEQ_SRC  = nbody.c
CUDA_SRC = nbody_cuda.cu

OMP_OBJ  = $(OMP_SRC:.c=.o)
SEQ_OBJ  = $(SEQ_SRC:.c=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)

# ================== EXECUTABLES ==================

OMP_EXE  = omp_nbody
SEQ_EXE  = seq_nbody
CUDA_EXE = cuda_nbody

# ================== DEFAULT ==================

all: $(OMP_EXE) $(SEQ_EXE) $(CUDA_EXE)

# ================== OPENMP BUILD ==================

$(OMP_EXE): $(OMP_OBJ)
	$(CC) $(COMMON_FLAGS) $(OMP_FLAGS) $^ $(LIBS) -o $@

$(OMP_OBJ): $(OMP_SRC)
	$(CC) $(COMMON_FLAGS) $(OMP_FLAGS) -c $< -o $@

# ================== SEQUENTIAL BUILD ==================

$(SEQ_EXE): $(SEQ_OBJ)
	$(CC) $(COMMON_FLAGS) $^ $(LIBS) -o $@

$(SEQ_OBJ): $(SEQ_SRC)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

# ================== CUDA BUILD ==================

$(CUDA_EXE): $(CUDA_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LIBS) -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# ================== RUN TARGETS ==================

run-omp: $(OMP_EXE)
	./$(OMP_EXE)

run-seq: $(SEQ_EXE)
	./$(SEQ_EXE)

run-cuda: $(CUDA_EXE)
	./$(CUDA_EXE)

# ================== CLEAN ==================

clean:
	rm -f $(OMP_EXE) $(SEQ_EXE) $(CUDA_EXE) $(OMP_OBJ) $(SEQ_OBJ) $(CUDA_OBJ)

.PHONY: all run-omp run-seq run-cuda clean