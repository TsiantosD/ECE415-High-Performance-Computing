# ==========================================
# C Makefile for CLAHE Project
# ==========================================

# 1. Compiler settings
CC := gcc

# 2. Target executable name
TARGET := main

# 3. Source files
# List all .c files here so they compile together
SRCS := main.c clahe.c helpers.c

# 4. Object files
# Automatically converts main.c clahe.c -> main.o clahe.o
OBJS := $(SRCS:.c=.o)

# 5. Base Flags (Always used)
# -Wall -Wextra: specific warning flags to catch errors
CFLAGS := -Wall -Wextra

# ------------------------------------------
# Build Rules
# ------------------------------------------

# --- 1. Default Build (Release) ---
# Usage: make
# Adds -O2 for speed, no debug info
all: CFLAGS += -O3
all: $(TARGET)

# --- 2. Debug Build ---
# Usage: make debug
# Adds -g (debug info) and -O0 (disables optimization for clearer debugging)
debug: CFLAGS += -g -O0
debug: $(TARGET)

# --- Link Step ---
# Creates the executable from the object files
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# --- Compile Step ---
# Compiles each .c file into a .o file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# --- Clean ---
clean:
	rm -f $(TARGET) $(OBJS)