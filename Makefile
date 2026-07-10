CC = gcc
CFLAGS = -O2 -Iinclude
LDFLAGS = -lm

ARCH := $(shell uname -m)

ifeq ($(ARCH),aarch64)
    ARCH_DEFINE = -DARCH_AARCH64
    KERNEL_SRC = src/kernels/aarch64/matmul_neon64.s \
                 src/kernels/aarch64/rmsnorm_neon64.s \
                 src/kernels/aarch64/state_update_neon64.s \
                 src/kernels/aarch64/dot_accum_neon64.s \
                 src/kernels/aarch64/vecmul_neon64.s \
                 src/kernels/aarch64/softmax_neon64.s
else ifeq ($(ARCH),x86_64)
    ARCH_DEFINE = -DARCH_X86_64
    KERNEL_SRC = src/kernels/x86_64/matmul_avx2.s \
                 src/kernels/x86_64/rmsnorm_avx2.s \
                 src/kernels/x86_64/state_update_avx2.s \
                 src/kernels/x86_64/dot_accum_avx2.s \
                 src/kernels/x86_64/vecmul_avx2.s \
                 src/kernels/x86_64/softmax_avx2.s
else
    ARCH_DEFINE =
    KERNEL_SRC =
endif

CORE_SRC = \
    src/tensor/tensor_alloc.c \
    src/tensor/tensor_ops.c \
    src/model/retention_multihead.c \
    src/model/retention_parallel.c \
    src/model/retention_recurrent.c \
    src/model/norm.c \
    src/model/ffn.c \
    src/model/model.c \
    src/autograd/backward_ops.c \
    src/train/model_train.c \
    src/train/loss.c \
    src/optimizer/adafactor.c \
    src/tokenizer/bpe_train.c \
    src/tokenizer/bpe_encode.c \
    src/format/pr_file.c \
    src/format/json_parser.c \
    src/kernels/dispatch.c

CLI_SRC = \
    src/cli/main.c \
    src/cli/cmd_build.c \
    src/cli/cmd_train.c \
    src/cli/cmd_delete.c

ALL_SRC = $(CORE_SRC) $(CLI_SRC) $(KERNEL_SRC)

BIN = bin/pointretta

.PHONY: all clean info

all: $(BIN)

$(BIN): $(ALL_SRC)
	mkdir -p bin
	$(CC) $(CFLAGS) $(ARCH_DEFINE) -o $(BIN) $(ALL_SRC) $(LDFLAGS)

info:
	@echo "Detected arch: $(ARCH)"
	@echo "Arch define:   $(ARCH_DEFINE)"
	@echo "Kernel source: $(KERNEL_SRC)"

clean:
	rm -f $(BIN)
