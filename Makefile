CXX := clang++
CXXFLAGS := -std=c++20 -O3 -march=armv8.2-a+simd+dotprod -Wno-unknown-pragmas
TARGET := bench
SRC := main.cpp

MODEL ?= stories110M.bin
TOKENIZER ?= tokenizer.bin

all: $(TARGET)

$(TARGET): $(SRC) inference_engine.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) benchmark_results.csv performance_figure.png

run: $(TARGET)
	./$(TARGET) $(MODEL) $(TOKENIZER)

.PHONY: all clean run
