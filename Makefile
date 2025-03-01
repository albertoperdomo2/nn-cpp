.PHONY: setup build clean help run

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the project
	@echo "Building project..."
	@mkdir -p build && cd build && cmake .. && make
	@if [ -f "build/mnist" ]; then \
		echo "Build successful! Run the demo with: ./build/mnist after running: make setup"; \
	elif [ -f "build/src/mnist" ]; then \
		echo "Build successful! Run the demo with: ./build/src/mnist after running: make setup"; \
	else \
		echo "Build completed, but mnist executable not found at expected location."; \
	fi

run: build  ## Build and run the mnist application
	@if [ -f "build/mnist" ]; then \
		./build/mnist; \
	elif [ -f "build/src/mnist" ]; then \
		./build/src/mnist; \
	else \
		echo "Error: mnist executable not found after build"; \
		exit 1; \
	fi

setup:  ## Setup project
	@echo "Setting up project..."
	@cd data && ./download.sh

clean:  ## Clean up files
	@echo "Cleaning up..."
	@if [ -d "data" ]; then \
		cd data/ && rm -rf t10k-images.idx3-ubyte t10k-labels.idx1-ubyte train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte; \
	fi
	@rm -rf build
	@rm -rf test/build

.DEFAULT_GOAL := help
