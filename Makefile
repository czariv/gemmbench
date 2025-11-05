# Compiler, Assembler, Debugger
CC := clang
AS := clang
GDB := gdb

# Default directories
ifeq ($(DRIVERDIR),)
	DRIVERDIR := ./driver
endif

# Targets
TARGET_EXE = gemmvbench
TARGET_LIB = libgemm.so

# Determine if kernel directory is needed
ifneq ($(filter run gdb distclean,$(MAKECMDGOALS)),)
    NEED_KERNELDIR := no
else
    NEED_KERNELDIR := yes
endif

# Load parameters from JSON
ifeq ($(NEED_KERNELDIR),yes)
	ifeq ($(KERNELDIR),)
        	$(error KERNELDIR is required. Usage: make <target> KERNELDIR=<path/to/kernel_dir>)
	else
		CONFIGFILE := $(KERNELDIR)/parameter.json

		# Extract parameters using jq
		TYPE := $(shell jq -r '.params.TYPE' $(CONFIGFILE))
		BUFFER_SIZE := $(shell jq -r '.params.BUFFER_SIZE' $(CONFIGFILE))
		EVAL_THRESHOLD := $(shell jq -r '.params.EVAL_THRESHOLD' $(CONFIGFILE))
		GEMM_UNROLL_M := $(shell jq -r '.kernel.GEMM_UNROLL_M' $(CONFIGFILE))
		GEMM_UNROLL_N := $(shell jq -r '.kernel.GEMM_UNROLL_N' $(CONFIGFILE))
		GEMM_P := $(shell jq -r '.tiling.GEMM_P' $(CONFIGFILE))
		GEMM_Q := $(shell jq -r '.tiling.GEMM_Q' $(CONFIGFILE))
		GEMM_R := $(shell jq -r '.tiling.GEMM_R' $(CONFIGFILE))

		CFLAGS += -fPIC -g -O3 -march=skylake-avx512 -Wall -Wextra \
			-I${DRIVERDIR} -I${KERNELDIR} \
			-DTYPE=$(TYPE) \
			-DBUFFER_SIZE=$(BUFFER_SIZE) \
			-DEVAL_THRESHOLD=$(EVAL_THRESHOLD) \
			-DGEMM_UNROLL_M=$(GEMM_UNROLL_M) \
			-DGEMM_UNROLL_N=$(GEMM_UNROLL_N) \
			-DGEMM_P=$(GEMM_P) \
			-DGEMM_Q=$(GEMM_Q) \
			-DGEMM_R=$(GEMM_R)

		ifeq ($(DEBUG), 1)
			CFLAGS += -DTIME
		else ifeq ($(DEBUG), 2)
			CFLAGS += -DTIME -DDEBUG
		else ifeq ($(DEBUG), 3)
			CFLAGS += -DTIME -DPERF -DDEBUG
		endif

		# Source files for driver and kernel
		COMMON_SRCS = ${DRIVERDIR}/interface.c \
			${DRIVERDIR}/level3.c \
			$(KERNELDIR)/gemm_icopy.c \
			$(KERNELDIR)/gemm_ocopy.c \
			$(KERNELDIR)/gemm_kernel.c \
			$(KERNELDIR)/gemm_beta.c \
			$(KERNELDIR)/generic_icopy.c \
			$(KERNELDIR)/generic_ocopy.c

		ifneq ($(SEQ),)
			COMMON_SRCS += \
				${DRIVERDIR}/interface_seq.c \
				${DRIVERDIR}/level3_seq.c \
				$(KERNELDIR)/gemm_kernel_pre.c
		endif
	endif
endif

# Choose build mode
ifeq ($(MODE),lib)
	TARGET := $(TARGET_LIB)
	SRCS := $(COMMON_SRCS)
else ifeq ($(MODE),exe)
	TARGET := $(TARGET_EXE)
	ifneq ($(SEQ),)
		SRCS := ./main_sequential.c $(COMMON_SRCS)
	else
		SRCS := ./main.c $(COMMON_SRCS)
	endif
else
	# Default to executable mode
	TARGET := $(TARGET_EXE)
	ifneq ($(SEQ),)
		SRCS := ./main_sequential.c $(COMMON_SRCS)
	else
		SRCS := ./main.c $(COMMON_SRCS)
	endif
endif

OBJS := $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Build shared library
lib: MODE=lib
lib: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

# Build executable
exe: MODE=exe
exe: $(TARGET_EXE)

$(TARGET_EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Assemble .s sources
%.o: %.s
	$(AS) $(CFLAGS) -c $< -o $@

# Run
run:
	./$(TARGET_EXE) 512 512 512 1 1 && echo "Success" || echo "Failure"

# Debug
gdb:
	$(GDB) --args $(TARGET_EXE) 16 24 256 1 1

# Clean
clean:
	-rm -f $(OBJS) 1>/dev/null 2>&1

distclean:
	$(MAKE) clean
	-rm -f $(TARGET_EXE) $(TARGET_LIB) 1>/dev/null 2>&1

.PHONY: all lib exe run gdb clean distclean
