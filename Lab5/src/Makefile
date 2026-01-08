# ================== COMPILERS ==================

CC = gcc
NVCC = nvcc

# ================== FLAGS ==================

# Common optimization flags for CPU (GCC)
COMMON_FLAGS = -Wall -O3 -fopenmp -ffast-math -lm
# LDFLAGS = -lnvToolsExt
# CUDA flags
# -Xcompiler passes flags to the host compiler (gcc). 
# We pass -fopenmp so the host side of the CUDA code can spawn threads.
# --use_fast_math is the CUDA equivalent of -ffast-math
# useful --ptxas-options=-v 
NVCC_FLAGS = -O3 --ptxas-options=-v --use_fast_math -Xcompiler -fopenmp

# Mode for main target
CPU_MODE ?= omp

# Add user flags
COMMON_FLAGS += $(USER_FLAGS)
NVCC_FLAGS   += $(USER_FLAGS)

# ================== SOURCES ==================

MAIN_SRC  = main.c
RESULTS_CHECK_SRC  = results_check.c
OMP_SRC  = nbody_omp.c
SEQ_SRC  = nbody.c
CUDA_SRC ?= nbody_cuda.cu

MAIN_OBJ = $(MAIN_SRC:.c=.o)
RESULTS_CHECK_OBJ = $(RESULTS_CHECK_SRC:.c=.o)
OMP_OBJ  = $(OMP_SRC:.c=.o)
SEQ_OBJ  = $(SEQ_SRC:.c=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)

ifeq ($(CPU_MODE), seq)
    CHOSEN_CPU_SRC = $(SEQ_SRC)
    CHOSEN_CPU_OBJ = $(CHOSEN_CPU_SRC:.c=.o)
    CPU_COMPILE_FLAGS = $(COMMON_FLAGS)
else
    CHOSEN_CPU_SRC = $(OMP_SRC)
    CHOSEN_CPU_OBJ = $(CHOSEN_CPU_SRC:.c=.o)
    CPU_COMPILE_FLAGS = $(COMMON_FLAGS) $(OMP_FLAGS)
endif


# ================== CPU-ONLY MODE ==================

ifeq ($(ONLY_CPU),1)
    CUDA_OBJ :=
    LINKER := $(CC)
    LINK_FLAGS := $(COMMON_FLAGS)
else
    LINKER := $(NVCC)
    LINK_FLAGS := $(NVCC_FLAGS)
endif


# ================== EXECUTABLES ==================

MAIN_EXE = main
RESULTS_CHECK_EXE = results_check

# ================== DEFAULT ==================

all: $(MAIN_EXE) $(RESULTS_CHECK_EXE)

debug: NVCC_FLAGS += --ptxas-options="-v" -g -G -O0
debug: $(MAIN_EXE) $(RESULTS_CHECK_EXE)

# ================== MAIN TARGET ==================

$(MAIN_EXE): $(MAIN_OBJ) $(CHOSEN_CPU_OBJ) $(CUDA_OBJ)
	$(LINKER) $(LINK_FLAGS) $^ -o $@ $(LDFLAGS)

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

# ================== MAIN_CPU_CHECK TARGET ==================

$(RESULTS_CHECK_EXE): $(RESULTS_CHECK_OBJ)
	$(CC) $(COMMON_FLAGS) $^ -o $@

$(RESULTS_CHECK_OBJ): $(RESULTS_CHECK_SRC)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

# ================== OPENMP BUILD ==================

$(OMP_OBJ): $(OMP_SRC)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

# ================== SEQUENTIAL BUILD ==================

$(SEQ_OBJ): $(SEQ_SRC)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

# ================== CUDA BUILD ==================

ifneq ($(ONLY_CPU),1)
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

# ================== CLEAN ==================

clean:
	rm -f $(MAIN_EXE) $(RESULTS_CHECK_EXE) *.o

.PHONY: all clean
