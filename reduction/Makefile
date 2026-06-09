# CUDA Makefile

# Compiler and flags
NVCC := nvcc -arch=sm_60
CXXFLAGS := -Iinclude/ -std=c++17 -DNO_CPU -DNO_VANILLA -DNO_THREAD_LINEAR -DNO_SHARED -DNO_SHARED_LINEAR -DNO_WARP_UNROLL #-DNO_FULL_UNROLL

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Test 
NUM_TEST := 10
PROBLEM_SIZE := (1 << 22)

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

run: $(TARGET)
	@VAL=$$(( $(PROBLEM_SIZE) )); \
	./$(TARGET) $$VAL

run_test: $(TARGET)
	@echo "Evaluating problem size and running tests..."
	@VAL=$$(( $(PROBLEM_SIZE) )); \
	if [ $$(( VAL % 1048576 )) -eq 0 ]; then \
		DIR_NAME="$$((VAL / 1048576))M"; \
	elif [ $$(( VAL % 1024 )) -eq 0 ]; then \
		DIR_NAME="$$((VAL / 1024))K"; \
	else \
		DIR_NAME="$$VAL"; \
	fi; \
	OUT_DIR="./profile_data/$$DIR_NAME"; \
	mkdir -p "$$OUT_DIR"; \
	i=0; \
	while [ $$i -lt $(NUM_TEST) ]; do \
		echo "Run $$i: Executing and moving profile_summary.csv to $$OUT_DIR/$${i}_profile_summary.csv"; \
		./$(TARGET) $$VAL; \
		if [ -f ./profile_summary.csv ]; then \
			mv ./profile_summary.csv "$$OUT_DIR/$${i}_profile_summary.csv"; \
		else \
			echo "Error: profile_summary.csv was not generated on run $$i."; \
			exit 1; \
		fi; \
		i=$$((i+1)); \
	done
	@echo "All tests completed successfully." 

.PHONY: all clean run run_test
