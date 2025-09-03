# Compiler, Simmulator and debbuger
CC := clang
AS := clang
GDB := gdb

# Compiler and linker flags
# Read values from JSON
GEMM_UNROLL_M=$(shell jq -r 'kernel.gemm_unroll_m' $(CONFIG_FILE))
GEMM_UNROLL_N=$(shell jq -r 'kernel.gemm_unroll_n' $(CONFIG_FILE))
GEMM_P=$(shell jq -r 'tiling.gemm_p' $(CONFIG_FILE))
GEMM_Q=$(shell jq -r 'tiling.gemm_q' $(CONFIG_FILE))
GEMM_R=$(shell jq -r 'tiling.gemm_r' $(CONFIG_FILE))

# Pass them as compiler flags
LDFLAGS += -static 

ifneq ($(filter run gdb distclean,$(MAKECMDGOALS)),)
    NEED_KERNELDIR := no
else
    NEED_KERNELDIR := yes
endif

ifeq ($(NEED_KERNELDIR),yes)
	ifeq ($(KERNELDIR),)
        	$(error KERNELDIR is required. Usage: make <target> KERNELDIR=<file.json>)
	else
		CONFIGFILE := $(KERNELDIR)/parameter.json

		# Extract parameters from JSON using jq
		TYPE := $(shell jq -r '.params.TYPE' $(CONFIGFILE))
		BUFFER_SIZE := $(shell jq -r '.params.BUFFER_SIZE' $(CONFIGFILE))
		EVAL_THRESHOLD := $(shell jq -r '.params.EVAL_THRESHOLD' $(CONFIGFILE))
		GEMM_UNROLL_M := $(shell jq -r '.kernel.GEMM_UNROLL_M' $(CONFIGFILE))
		GEMM_UNROLL_N := $(shell jq -r '.kernel.GEMM_UNROLL_N' $(CONFIGFILE))
		GEMM_P := $(shell jq -r '.tiling.GEMM_P' $(CONFIGFILE))
		GEMM_Q := $(shell jq -r '.tiling.GEMM_Q' $(CONFIGFILE))
		GEMM_R := $(shell jq -r '.tiling.GEMM_R' $(CONFIGFILE))

		CFLAGS += -g -O3 -march=skylake-avx512  -Wall -Wextra -I./driver -I./kernel/riscv64 \
				-DTYPE=$(TYPE) \
				-DBUFFER_SIZE=$(BUFFER_SIZE) \
				-DEVAL_THRESHOLD=$(EVAL_THRESHOLD) \
				-DGEMM_UNROLL_M=$(GEMM_UNROLL_M) \
				-DGEMM_UNROLL_N=$(GEMM_UNROLL_N) \
				-DGEMM_P=$(GEMM_P) \
				-DGEMM_Q=$(GEMM_Q) \
				-DGEMM_R=$(GEMM_R)

		SRCS =  main.c \
			driver/interface.c \
			driver/level3.c \
			$(KERNELDIR)/gemm_icopy.c \
			$(KERNELDIR)/gemm_ocopy.c \
			$(KERNELDIR)/gemm_kernel.c \
			$(KERNELDIR)/gemm_beta.c

		OBJS = $(SRCS:.c=.o) 
	endif
endif

# Executable name
TARGET = gemmvbench

# Build executable
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIB_NAME) $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(CFLAGS) -c $< -o $@

# Assemble .s files into object files
%.o: %.s
	$(AS) $(CFLAGS) -c $< -o $@

# Run with QEMU
run:
	./$(TARGET) 512 512 512 1 1 && echo "Success" || echo "Failure"

# Run with QEMU and GDB
gdb:
	$(GDB) --args $(TARGET) 16 24 256 1 1

# Clean build files
clean:
	-rm -f $(OBJS) 1>/dev/null 2>&1

distclean:
	$(MAKE) clean
	-rm -f $(TARGET) 1>/dev/null 2>&1

.PHONY: all run gdb clean distclean

