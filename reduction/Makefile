# CUDA Makefile

# Compiler and flags
NVCC := nvcc
CXXFLAGS := -Iinclude/ -std=c++17 -DNO_CPU #-DNO_VANILLA -DNO_THREAD_LINEAR -DNO_SHARED

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))
TARGET := $(BIN_DIR)/reduction

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean