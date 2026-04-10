NVCC = nvcc
ARCH = -arch=sm_75

SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%)

all: $(BUILD_DIR) $(OBJS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(ARCH) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
