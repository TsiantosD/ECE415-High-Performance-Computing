# ================== COMPILERS ==================

CC = gcc

# ================== FLAGS ==================

# Common optimization flags (safe for both)
COMMON_FLAGS = -Wall -O3 -march=native -ffast-math

# OpenMP flags (ONLY for parallel build)
OMP_FLAGS = -fopenmp

# Math library
LIBS = -lm

# ================== SOURCES ==================

OMP_SRC = nbody_omp.c
SEQ_SRC = nbody.c

OMP_OBJ = $(OMP_SRC:.c=.o)
SEQ_OBJ = $(SEQ_SRC:.c=.o)

# ================== EXECUTABLES ==================

OMP_EXE = omp_nbody
SEQ_EXE = seq_nbody

# ================== DEFAULT ==================

all: $(OMP_EXE) $(SEQ_EXE)

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

# ================== RUN TARGETS ==================

run-omp: $(OMP_EXE)
	./$(OMP_EXE)

run-seq: $(SEQ_EXE)
	./$(SEQ_EXE)

# ================== CLEAN ==================

clean:
	rm -f $(OMP_EXE) $(SEQ_EXE) $(OMP_OBJ) $(SEQ_OBJ)

.PHONY: all run-omp run-seq clean
