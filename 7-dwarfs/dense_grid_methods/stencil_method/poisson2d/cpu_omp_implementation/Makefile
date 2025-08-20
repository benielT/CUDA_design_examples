CXX = g++

CURRENT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(CURRENT_DIR)

CXXFLAGS = -I$(BASE_DIR)/include -std=c++17 -fopenmp
LDFLAGS = -pthread

SRC = $(wildcard $(BASE_DIR)/src/*.cpp)
TEST_SRC = $(wildcard $(BASE_DIR)/test/*.cpp)

OBJ = $(SRC:.cpp=.o)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

TARGET = $(BASE_DIR)/poisson2d_omp
TEST_TARGET = $(BASE_DIR)/poisson2d_omp_test

.PHONY: all clean test run_experiment

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

$(BASE_DIR)/src/%.o: $(BASE_DIR)/src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BASE_DIR)/test/%.o: $(BASE_DIR)/test/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<


run_experiment: $(TARGET)
	chmod +x $(BASE_DIR)/run_experiment.sh
	cd $(BASE_DIR) && ./run_experiment.sh

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TARGET) $(TEST_TARGET)
