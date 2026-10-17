# Makefile for par2afft with automatic dependency detection
#
# Builds the following things:
#
#   build/libpar2afft.a:
#       a static library consisting of all source files that do not contain an
#       implementaiton of main(). This library is included in all the binaries
#       built below.
#
#   build/par2afft:
#       the main binary which provides the par2afft CLI tool
#
#   build/*_test:
#       test binaries (can be run individually or via `make test`)
#
#   build/bench*:
#       benchmarks (can be run individually or via `make bench`)
#

# Directories where to store binaries, object files (.o) and dependency files (.d)
# Note that they can be the same directory.
OUTDIR=build
OBJDIR=build
DEPDIR=build

# Compiler and linker flags
CPPFLAGS=-D_POSIX_C_SOURCE=200809 -D_FILE_OFFSET_BITS=64
CFLAGS=-Wall -Wno-deprecated-declarations -std=c11 -O3 -march=x86-64-v3 -g
LDLIBS=
LIBS=$(OUTDIR)/libpar2afft.a

# Build with `make HAVE_OPENSSL=1` to use OpenSSL implementation of MD5.
# Otherwise, the vendored implementation by Solar Designer is used instead.
ifeq ($(HAVE_OPENSSL), 1)
LDLIBS+=-lcrypto
CPPFLAGS+=-DHAVE_OPENSSL
else
LIBS+=$(OUTDIR)/libmd5.a
endif

# Sources to include in libpar2afft.a (essentially everything that is not a binary
# source; i.e., does not contain a definition of main())
LIBPAR2AFFT_SRCS= \
	crc32.c \
	gf16.c \
	gf16_avx2_mul.c \
	gf16_vt_batch_mul.c \
	gf16_vt_mul.c

# Main binary name. Source is in par2afft.c
MAIN_NAME=par2afft

# Test binaries. Each of these must have a corresponding .c file.
TEST_NAMES= \
	cantor_basis_test \
	crc32_test \
	gf16_avx2_mul_test \
	gf16_vt_batch_mul_test \
	gf16_vt_mul_test

# Benchmark binaries. Each of these must have a corresponding .c file.
BENCH_NAMES= \
	gf16_avx2_mul_bench \
	gf16_vt_batch_mul_bench \
	gf16_vt_mul_bench \
	md5_bench

#
# These rest of the file is mostly boilerplate.
#

MAIN_BINARY=$(OUTDIR)/$(MAIN_NAME)
TEST_BINARIES=$(addprefix $(OUTDIR)/,$(TEST_NAMES))
BENCH_BINARIES=$(addprefix $(OUTDIR)/,$(BENCH_NAMES))
ALL_BINARIES=$(MAIN_BINARY) $(TEST_BINARIES) $(BENCH_BINARIES)

all: $(ALL_BINARIES)

clean:
	rm -f \
	    $(OBJDIR)/*.o \
	    $(DEPDIR)/*.d \
	    $(OUTDIR)/libpar2afft.a \
	    $(OUTDIR)/libmd5.a \
	    $(ALL_BINARIES)

test: $(TEST_BINARIES)
	./run-tests.sh $^

bench: $(BENCH_BINARIES)
	./run-tests.sh $^

.PHONY: all clean test bench

# Keep all intermediate files. Without this, gmake auto-deletes intermediates
# like the .d files that we need to track dependencies.
.NOTINTERMEDIATE:

# How to build the static library libpar2afft.a
$(OUTDIR)/libpar2afft.a: $(LIBPAR2AFFT_SRCS:%.c=$(OBJDIR)/%.o)
	ar crs $@ $^

$(OUTDIR)/md5.o: vendor/md5/md5.c vendor/md5/md5.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I vendor/md5 -c $< -o $@

$(OUTDIR)/libmd5.a: $(OUTDIR)/md5.o
	ar crc $@ $^

# How to build binaries (linking with libpar2afft.a)
$(OUTDIR)/%: $(OUTDIR)/%.o $(LIBS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# How to build objects from sources, and rebuild when dependencies change.
$(DEPDIR)/%.d: %.c
	$(CC) -MM -MP -MT $(OBJDIR)/$*.o $(CPPFLAGS) $< -o $@

$(OBJDIR)/%.o: %.c $(DEPDIR)/%.d
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

include $(wildcard $(DEPDIR)/*.d)
