NVCC = nvcc
ARCH = -arch=sm_75

SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

all: $(BUILD_DIR)/main

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# compile step
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(ARCH) -c $< -o $@

# link step
$(BUILD_DIR)/main: $(OBJS)
	$(NVCC) $(ARCH) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)
