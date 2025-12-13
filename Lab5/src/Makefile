# --- Configuration Variables ---

# Compiler to use
CC = gcc

# Standard compilation flags:
# -Wall: Enable all standard warnings
# -Wextra: Enable extra warnings (highly recommended)
# -std=c99 or -std=gnu99: Use the C99 standard (or GNU extension)
# -O3: Aggressive optimization level
# -fopenmp: Enable OpenMP support
# -lm: Link the math library (for sqrtf)
# Note: Since the timer code is platform-dependent, ensure you are compiling on the
# intended platform (e.g., Linux/Unix or Windows).
CFLAGS = -Wall -O3 -march=native -ffast-math -fopenmp -lm

# List of source files. ONLY nbody.c is a source file now, as the timer implementation is
# included directly via the timer.h header file.
SOURCES = nbody.c

# List of object files (derived from sources)
OBJECTS = $(SOURCES:.c=.o)

# Name of the final executable
EXECUTABLE = nbody_omp

# --- Targets ---

# Default target: builds the executable
all: $(EXECUTABLE)

# Rule to link the final executable
# This only needs to link the nbody.o file.
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

# Rule to compile C source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to run the program with a default number of bodies/systems
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Target to clean up generated files
clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

# Prevent make from confusing targets with file names
.PHONY: all run clean