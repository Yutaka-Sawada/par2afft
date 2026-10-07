// A minimal implementation of the par2 client for testing purposes.
//
// Uses the Additive FFT to multiply with the transpose Vandermonde matrix in
// O(q log q) time, where q is the size of the finite field, 65536 for Par2.

#ifdef _MSC_VER
// erase warning on Microsoft Visual Studio
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "crc32.h"
#include "gf16.h"
#include "gf16_vt_mul.h"
#include "gf16_vt_batch_mul.h"

#ifdef _MSC_VER // Windows

// MD5 hash generator -- Paul Houle (paulhoule.com) 11/13/2017
#include "win/phmd5.h"

// for low level IO access
#include <io.h>

/*
header-only Windows implementation of the `<unistd.h>` header.
MIT License
Copyright (c) 2019 win32ports
*/
#include "win/unistd.h"

#else // Linux
#include <openssl/md5.h>

#include <sys/mman.h>
#include <unistd.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#define CRC32_LENGTH 4

#define MAX_BLOCK_COUNT 32768
#define MAX_FILE_COUNT MAX_BLOCK_COUNT

// Note: buffer size must be a multiple of 4.
static const char creator_string[16] = "par2afft 0.0";

static const char *packet_type_creator  = "PAR 2.0\0Creator\0";
static const char *packet_type_main     = "PAR 2.0\0Main\0\0\0\0";
static const char *packet_type_filedesc = "PAR 2.0\0FileDesc";
static const char *packet_type_ifsc     = "PAR 2.0\0IFSC\0\0\0\0";
static const char *packet_type_recovery = "PAR 2.0\0RecvSlic";

#ifdef _MSC_VER // Windows

// set alignment to 1
#pragma pack(push, 1)

struct Md5 {
    uint8_t data[MD5_DIGEST_LENGTH];  // must be unsigned for compare_md5
};

struct Crc32 {
    uint8_t data[CRC32_LENGTH];
};

// Just blindly assuming structs are packed here.
// Also we assume we're on a little-endian system.
struct PacketHeader {
    uint8_t    magic[8];
    uint64_t   length;   /* must be multiple of 4 */
    struct Md5 md5;
    struct Md5 recovery_set_id;
    uint8_t    type[16];
};

struct MainPacketBody {
    uint64_t slice_size;
    uint32_t file_count;
    struct Md5 file_ids[MAX_FILE_COUNT];
};

struct MainPacket {
    struct PacketHeader header;
    struct MainPacketBody body;
};

struct FileDescriptionPacketBody {
    struct Md5 id;
    struct Md5 md5;
    struct Md5 md5_16k;
    uint64_t length;
    char     name[];
};

struct FileDescriptionPacket {
    struct PacketHeader header;
    struct FileDescriptionPacketBody body;
};

struct SliceChecksums {
    struct Md5   md5;
    struct Crc32 crc32;
};

// Input File Slice Checksums
struct IfscPacketBody {
    struct Md5 file_id;
    struct SliceChecksums checksums[];
};

struct IfscPacket {
    struct PacketHeader header;
    struct IfscPacketBody body;
};

struct RecoverySlicePacketBody {
    uint32_t exponent;
    gf16_t elems[];
};

struct RecoverySlicePacket {
    struct PacketHeader header;
    struct RecoverySlicePacketBody body;
};

// end alignment
#pragma pack(pop)

#else // Linux

struct Md5 {
    uint8_t data[MD5_DIGEST_LENGTH];  // must be unsigned for compare_md5
} __attribute__((packed));

struct Crc32 {
    uint8_t data[CRC32_LENGTH];
};

// Just blindly assuming structs are packed here.
// Also we assume we're on a little-endian system.
struct PacketHeader {
    uint8_t    magic[8];
    uint64_t   length;   /* must be multiple of 4 */
    struct Md5 md5;
    struct Md5 recovery_set_id;
    uint8_t    type[16];
} __attribute__((packed));

struct MainPacketBody {
    uint64_t slice_size;
    uint32_t file_count;
    struct Md5 file_ids[MAX_FILE_COUNT];
} __attribute__((packed));

struct MainPacket {
    struct PacketHeader header;
    struct MainPacketBody body;
} __attribute__((packed));

struct FileDescriptionPacketBody {
    struct Md5 id;
    struct Md5 md5;
    struct Md5 md5_16k;
    uint64_t length;
    char     name[];
} __attribute__((packed));

struct FileDescriptionPacket {
    struct PacketHeader header;
    struct FileDescriptionPacketBody body;
} __attribute__((packed));

struct SliceChecksums {
    struct Md5   md5;
    struct Crc32 crc32;
} __attribute__((packed));

// Input File Slice Checksums
struct IfscPacketBody {
    struct Md5 file_id;
    struct SliceChecksums checksums[];
} __attribute__((packed));

struct IfscPacket {
    struct PacketHeader header;
    struct IfscPacketBody body;
} __attribute__((packed));

struct RecoverySlicePacketBody {
    uint32_t exponent;
    gf16_t elems[];
} __attribute__((packed));

struct RecoverySlicePacket {
    struct PacketHeader header;
    struct RecoverySlicePacketBody body;
} __attribute__((packed));

#endif

struct InputFile {
    char    *name;
    uint64_t size;
    int      fd;
    void    *mapped_addr;
    size_t   mapped_size;
    struct Md5 md5_16k;
    struct Md5 id;
};

struct InputSlice {
    gf16_t        constant;
    const gf16_t *elems;
};

static size_t pagesize;

static struct InputFile input_files[MAX_FILE_COUNT];

static int input_file_count = 0;

static struct InputSlice input_blocks[MAX_BLOCK_COUNT];

static struct RecoverySlicePacket *recovery_blocks[MAX_BLOCK_COUNT];

static struct Md5 recovery_set_id;

static struct MainPacket main_packet;

static int out_fd = -1;
static void *out_mapped_addr = NULL;
static size_t out_mapped_size = 0;

static int64_t arg_input_blocks   =    -1;
static int64_t arg_block_size     =    -1;
static int64_t arg_redundancy     =    -1;
static int64_t arg_output_blocks  =    -1;
static int64_t arg_output_start   =     0;
static bool      arg_batch          = false;
static bool      arg_overwrite      = false;

static struct Crc32 wrap_crc32(uint32_t crc32) {
    // Store in little endian byte order.
    struct Crc32 res;
    res.data[0] = (crc32 >>  0) & 0xff;
    res.data[1] = (crc32 >>  8) & 0xff;
    res.data[2] = (crc32 >> 16) & 0xff;
    res.data[3] = (crc32 >> 24) & 0xff;
    return res;
}

static void debug_print_hex(const void *data, size_t size) {
    const unsigned char *p = data;
    while (size--) fprintf(stderr, "%02x", *p++);
    fprintf(stderr, "\n");
}

static void debug_print_md5(const struct Md5 *md5) {
    debug_print_hex(md5->data, MD5_DIGEST_LENGTH);
}

// Compare two MD5 hashes as if they were integers in little endian byte order,
// i.e., the last byte is the most significant one.
static int compare_md5(const struct Md5 *a, const struct Md5 *b) {
    for (int i = MD5_DIGEST_LENGTH - 1; i >= 0; --i) {
        int diff = a->data[i] - b->data[i];
        if (diff != 0) return diff;
    }
    return 0;
}

static void show_usage() {
    fputs(creator_string, stderr);
    fputs(
        "\n"
        "Usage: par2afft create [options] <output.par2> <input>...\n"
        "\n"
        "Options:\n"
        "    -b<n>   block count (default 2000)\n"
        "    -s<n>   block size (default auto)\n"
        "    -r<n>   redundancy percentage (default 5)\n"
        "    -c<n>   recovery block count (default 100)\n"
        "    -f<n>   first recovery block number (default 0)\n"
        "Note: space after options is not allowed.\n"
        "\n"
        "Experimental options:\n"
        "    --batch      use batch multiplication to calculate recovery blocks (faster)\n"
        "    --overwrite  overwrite the output file if it already exists\n",
        stderr);
}

static int parse_option(const char *arg) {
    if (strcmp(arg, "--batch") == 0) {
        arg_batch = true;
        return 0;
    }
    if (strcmp(arg, "--overwrite") == 0) {
        arg_overwrite = true;
        return 0;
    }
    switch (arg[1]) {
    case 'b':
        if (sscanf(&arg[2], "%lld", &arg_input_blocks) == 1) return 0;
        break;
    case 's':
        if (sscanf(&arg[2], "%lld", &arg_block_size) == 1) return 0;
        break;
    case 'r':
        if (sscanf(&arg[2], "%lld", &arg_redundancy) == 1) return 0;
        break;
    case 'c':
        if (sscanf(&arg[2], "%lld", &arg_output_blocks) == 1) return 0;
        break;
    case 'f':
        if (sscanf(&arg[2], "%lld", &arg_output_start) == 1) return 0;
        break;
    }
    fprintf(stderr, "Invalid option argument: %s\n", arg);
    return -1;
}

static void fill_packet_header(
        struct PacketHeader *header, const char packet_type[16],
        const void *body_data, size_t body_length) {
    assert(body_length % 4 == 0);
    memcpy(header->magic, "PAR2\0PKT", 8);
    header->length = body_length + sizeof(struct PacketHeader);
    header->recovery_set_id = recovery_set_id;
    memcpy(header->type, packet_type, 16);

    MD5_CTX md5_ctx;
    MD5_Init(&md5_ctx);
    MD5_Update(&md5_ctx, &header->recovery_set_id,
            sizeof(struct PacketHeader) - offsetof(struct PacketHeader, recovery_set_id));
    MD5_Update(&md5_ctx, body_data, body_length);
    MD5_Final(header->md5.data, &md5_ctx);
}

static int write_packet(struct PacketHeader *header, const void *body_data) {
    size_t header_length = sizeof(struct PacketHeader);
    assert(header->length >= header_length);
#ifdef _MSC_VER
    if (_write(out_fd, header, (int)header_length) != (int)header_length) {
#else
    if (write(out_fd, header, header_length) != header_length) {
#endif
        perror("Failed to write packet header");
        return -1;
    }
    size_t body_length = header->length - header_length;
#ifdef _MSC_VER
    if (_write(out_fd, body_data, (int)body_length) != (int)body_length) {
#else
    if (write(out_fd, body_data, body_length) != body_length) {
#endif
        perror("Failed to write packet body");
        return -1;
    }
    return 0;
}

static int write_creator_packet() {
    struct PacketHeader header;
    fill_packet_header(&header, packet_type_creator, creator_string, sizeof(creator_string));
    return write_packet(&header, creator_string);
}

static int write_main_packet() {
    return write_packet(&main_packet.header, &main_packet.body);
}

static int write_file_description_packet(const struct InputFile *input_file) {
    size_t name_len = strlen(input_file->name);
    if (name_len % 4) name_len += 4 - name_len % 4;  // pad to 4 bytes
    struct FileDescriptionPacket *packet =
        calloc(1, sizeof(struct FileDescriptionPacket) + name_len);
    if (packet == NULL) {
        perror("calloc");
        return -1;
    }
    packet->body.id = input_file->id;
    packet->body.md5_16k = input_file->md5_16k;
    packet->body.length = input_file->size;
    strncpy(packet->body.name, input_file->name, name_len);
    MD5_CTX md5_ctx;
    MD5_Init(&md5_ctx);
    MD5_Update(&md5_ctx, input_file->mapped_addr, input_file->size);
    MD5_Final(packet->body.md5.data, &md5_ctx);
    fill_packet_header(&packet->header, packet_type_filedesc,
            &packet->body, sizeof(struct FileDescriptionPacketBody) + name_len);
    int res = write_packet(&packet->header, &packet->body);
    free(packet);
    return res;
}

// This writes the IFSC packet and as a side-effect also populates input_slices.
static int write_ifsc_packet(const struct InputFile *input_file, struct InputSlice **slice_ptr) {
    const uint8_t *data = input_file->mapped_addr;
    int64_t size = input_file->size;
    uint64_t slices = (size + arg_block_size - 1) / arg_block_size;

    size_t payload_size = sizeof(struct SliceChecksums) * slices;
    struct IfscPacket *packet = calloc(1, sizeof(struct IfscPacket) + payload_size);
    if (packet == NULL) {
        perror("calloc");
        return -1;
    }
    packet->body.file_id = input_file->id;
    for (uint64_t i = 0; i < slices; ++i) {
        assert(size > 0);

        // Calculate CRC-32 of slice
        packet->body.checksums[i].crc32 = wrap_crc32(crc32(data, arg_block_size));

        // Calculate MD5 hash of slice
        MD5_CTX md5_ctx;
        MD5_Init(&md5_ctx);
        MD5_Update(&md5_ctx, data, arg_block_size);
        MD5_Final(packet->body.checksums[i].md5.data, &md5_ctx);

        (*slice_ptr)++->elems = (const gf16_t*) data;

        int64_t slice_size = size < arg_block_size ? size : arg_block_size;
        data += slice_size;
        size -= slice_size;
    }
    assert(size == 0);
    fill_packet_header(&packet->header, packet_type_ifsc,
            &packet->body, sizeof(struct IfscPacketBody) + payload_size);
    int res = write_packet(&packet->header, &packet->body);
    free(packet);
    return res;
}

// Creates the main packet after input files have been opened.
// This also generates the recovery_set_id, which is needed for any other packets.
static void create_main_packet() {
    main_packet.body.slice_size = arg_block_size;
    main_packet.body.file_count = input_file_count;
    for (int i = 0; i < input_file_count; ++i) {
        main_packet.body.file_ids[i] = input_files[i].id;
    }

    // Generate recovery_set_id as the hash of the body of the main packet.
    size_t body_length = sizeof(uint64_t) + sizeof(uint32_t) +
            (size_t) input_file_count * MD5_DIGEST_LENGTH;
    MD5_CTX md5_ctx;
    MD5_Init(&md5_ctx);
    MD5_Update(&md5_ctx, &main_packet.body, body_length);
    MD5_Final(recovery_set_id.data, &md5_ctx);

    // Now we can use it to fill in the header
    fill_packet_header(&main_packet.header, packet_type_main, &main_packet.body, body_length);
}

static int open_output_file(const char *filename) {
    if (strcmp(filename, "-") == 0) {
#ifdef _MSC_VER
        out_fd = _fileno(stdout);
#else
        out_fd = fileno(stdout);
#endif
        return 0;
    }
    int open_flags = O_RDWR | O_BINARY | O_CREAT | (arg_overwrite ? O_TRUNC : O_EXCL);
#ifdef _MSC_VER
    out_fd = _open(filename, open_flags, _S_IREAD | _S_IWRITE);
#else
    out_fd = open(filename, open_flags, 0644);
#endif
    if (out_fd == -1) {
        if (errno == EEXIST) {
            fprintf(stderr, "Output file already exists: %s\n", filename);
        } else {
            perror("Failed to open output file");
        }
        return -1;
    }
    return 0;
}

// Verifies that the filename is suitable for inclusion in a PAR2 file and
// returns NULL, or else returns an error message.
const char *check_filename(const char *name) {
    if (name == NULL) return "filename is NULL";
    size_t len = strlen(name);
    if (len == 0) return "filename is empty";
    if (name[0] == '/') return "filename is an absolute path";
    if (name[len - 1] == '/') return "filename refers to a directory";
    for (size_t i = 0;;) {
        size_t j = i;
        while (j < len && name[j] != '/') ++j;
        if (j - i == 0) {
            return "filename contains an empty path component";
        }
        if (j - i == 1 && name[i] == '.') {
            return "filename contains a \".\" path component";
        }
        if (j - i == 2 && name[i] == '.' && name[i + 1] == '.') {
            return "filename contains a \"..\" path component";
        }
        if (j == len) return NULL;
        i = j + 1;
    }
}

static int open_input_file(const char *filename) {
    const char *err = check_filename(filename);
    if (err != NULL) {
        fprintf(stderr, "Invalid input file name (%s): %s\n", filename, err);
        return -1;
    }

    if (input_file_count >= MAX_FILE_COUNT) {
        fprintf(stderr, "Too many input files!\n");
        return -1;
    }

    int fd = open(filename, O_RDONLY | O_BINARY);
    if (fd == -1) {
        fprintf(stderr, "Failed to open input file (%s): %s\n", filename, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Failed to stat input file (%s): %s\n", filename, strerror(errno));
        goto fail;
    }
    if (!S_ISREG(st.st_mode)) {
        if (S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Input file is a directory: %s\n", filename);
        } else {
            fprintf(stderr, "Input file not a regular file: %s\n", filename);
        }
        goto fail;
    }
    if (st.st_size < 0 || st.st_size > UINT64_MAX) {
        fprintf(stderr, "Invalid file size\n");
        goto fail;
    } else if (st.st_size == 0) {
        fprintf(stderr, "Warning: %s is an empty file!\n", filename);
    }

    char *name = strdup(filename);
    if (name == NULL) {
        perror("strdup");
        goto fail;
    }

    struct InputFile *file = &input_files[input_file_count++];
    file->name = name;
    file->size = st.st_size;
    file->fd   = fd;
    return 0;

fail:
    if (close(fd) != 0) perror("close");
    return -1;
}

static int mmap_input_files() {
    assert(arg_block_size > 0 && pagesize > 0);  // should be initialized by now
    for (int i = 0; i < input_file_count; ++i) {
        struct InputFile *file = &input_files[i];
        if (file->size == 0) continue;

        // Round up to slice size, then to page size.
        size_t size = file->size;
        if (size % arg_block_size != 0) size += arg_block_size - size % arg_block_size;
        if (size % pagesize != 0) size += pagesize - size % pagesize;

#ifdef _MSC_VER // Windows OS doesn't support larger mapping size than actual file size.
        // First allocate memory that cover each slice completely.
        void *addr = malloc(size);
        if (addr == NULL) {
            fprintf(stderr, "Failed to allocate memory of length %zu for input file (%s): %s\n",
                size, file->name, strerror(errno));
            return -1;
        }
        // Then read the entire file on the allocated memory.
        FILE *fp = _fdopen(file->fd, "rb");
        if (fp == NULL){
            fprintf(stderr, "Failed to open input file (%s): %s\n", file->name, strerror(errno));
            return -1;
        }
        size_t load_size = fread(addr, 1, file->size, fp);
        if (load_size != file->size){
            fprintf(stderr, "Failed to read input file (%s): %s\n", file->name, strerror(errno));
            fclose(fp);
            return -1;
        }
        fclose(fp);
        if (file->size < size)
            memset((char *)addr + file->size, 0, size - file->size); // zero padding the last slice
        file->mapped_addr = addr;
        file->mapped_size = size;
#else
        // First allocate zero pages that cover each slice completely.
        void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) {
            fprintf(stderr,
                "Failed to create zero mapping of length %zu for input file (%s): %s\n",
                size, file->name, strerror(errno));
            return -1;
        }
        // Then create the file mapping on top, which may not fully cover the last slice.
        void *new_addr = mmap(addr, file->size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file->fd, 0);
        if (new_addr == MAP_FAILED || new_addr != addr) {
            fprintf(stderr, "Failed to mmap input file (%s): %s\n", file->name,
                new_addr == MAP_FAILED ? strerror(errno) : "invalid address");
            munmap(addr, size);
            if (new_addr != MAP_FAILED) munmap(new_addr, file->size);
            return -1;
        }
        file->mapped_addr = addr;
        file->mapped_size = size;
#endif
    }
    return 0;
}

static void calculate_input_file_ids() {
    for (int i = 0; i < input_file_count; ++i) {
        struct InputFile *file = &input_files[i];

        // Calculate MD5 of the first 16KiB of the file.
        MD5_CTX md5_ctx;
        MD5_Init(&md5_ctx);
        MD5_Update(&md5_ctx, file->mapped_addr, file->size < 16384 ? file->size : 16384);
        MD5_Final(file->md5_16k.data, &md5_ctx);

        // Calculate file ID as MD5 has of md5_16k, size, and name.
        MD5_Init(&md5_ctx);
        MD5_Update(&md5_ctx, file->md5_16k.data, MD5_DIGEST_LENGTH);
        MD5_Update(&md5_ctx, &file->size, sizeof(file->size));
        MD5_Update(&md5_ctx, file->name, strlen(file->name));
        MD5_Final(file->id.data, &md5_ctx);
    }
}

static int reserve_output_slices(int slice_count) {
    size_t recovery_packet_size = sizeof(struct RecoverySlicePacket) + arg_block_size;
    assert(recovery_packet_size % 4 == 0);
    off_t current_pos = lseek(out_fd, 0, SEEK_CUR);
    if (current_pos == -1) {
        perror("Failed to determine output position (writing to a pipe?)");
        return -1;
    }
    uint64_t space_needed = recovery_packet_size * slice_count;
#ifdef _MSC_VER // MSVC doesn't support mmap() and posix_fallocate().
    out_mapped_addr = malloc(space_needed);
    if (out_mapped_addr == NULL) {
        perror("Failed to allocate memory for recovery packets");
        return -1;
    }
    out_mapped_size = space_needed;
    char *ptr = (char*) out_mapped_addr;
#else
    if (posix_fallocate(out_fd, current_pos, space_needed) != 0) {
        // No perror(), because posix_fallocate() does NOT set errno on failure
        fprintf(stderr, "Failed to allocate disk space for recovery slices!\n");
        return -1;
    }
    uint64_t new_size = current_pos + space_needed;
    out_mapped_addr = mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (out_mapped_addr == MAP_FAILED) {
        perror("Failed to mmap output file");
        return -1;
    }
    out_mapped_size = new_size;
    char *ptr = (char*) out_mapped_addr + current_pos;
#endif

    for (int i = 0; i < slice_count; ++i) {
        struct RecoverySlicePacket *packet = (struct RecoverySlicePacket*)ptr;
        ptr += recovery_packet_size;
        recovery_blocks[i] = packet;
        packet->body.exponent = (uint32_t)(arg_output_start + i);
    }
    return 0;
}

static int finalize_output_slices(int slice_count) {
    size_t body_size = sizeof(struct RecoverySlicePacketBody) + arg_block_size;
    assert(body_size % 4 == 0);
    for (int i = 0; i < slice_count; ++i) {
        fill_packet_header(&recovery_blocks[i]->header, packet_type_recovery, &recovery_blocks[i]->body, body_size);
    }

#ifdef _MSC_VER // write recovery packets on output file
    FILE *fp = _fdopen(out_fd, "r+");
    if (fp == NULL){
        fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
        return -1;
    }
    if (fwrite(out_mapped_addr, 1, out_mapped_size, fp) != out_mapped_size) {
        fprintf(stderr, "Failed to write output file: %s\n", strerror(errno));
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // relese memory
    free(out_mapped_addr);
    out_mapped_addr = NULL;
#endif

    return 0;
}

static void generate_recovery_data() {
    assert(arg_block_size % 4 == 0 && sizeof(gf16_t) == 2);
    gf16_t a[1 << 16];
    gf16_t y[1 << 16];
    memset(a, 0, sizeof(a));
    int last_progress = isatty(fileno(stderr)) ? -1 : 100;
    int64_t columns = arg_block_size / sizeof(gf16_t);
    for (int64_t j = 0; j < columns; ++j) {
        // Print progress
        int progress = (int)(j * 100 / columns);
        if (progress > last_progress) {
            fprintf(stderr, "%3d%%\r", progress);
            last_progress = progress;
        }
        for (int i = 0; i < arg_input_blocks; ++i) {
            // Load the j-th element from the i-th input block.
            const struct InputSlice *s = &input_blocks[i];
            a[s->constant] = s->elems[j];
        }
        gf16_vt_mul(y, a);
        for (int i = 0; i < arg_output_blocks; ++i) {
            struct RecoverySlicePacketBody *body = &recovery_blocks[i]->body;
            body->elems[j] = y[body->exponent];
        }
    }
}

// Batch size in elements (i.e. 256 = 512 KIB per batch).
#define AFFT_BATCH_SIZE 256

static gf16_t afft_buffer[GF16_ORDER][AFFT_BATCH_SIZE]; // 128 KiB * BATCH_SIZE

static void generate_recovery_data_batch() {
    const gf16_t *input_rows[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) input_rows[i] = NULL;

    gf16_t *output_rows[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) output_rows[i] = afft_buffer[i];

    assert(arg_block_size % sizeof(gf16_t) == 0);
    uint64_t total_elements = arg_block_size / sizeof(gf16_t);
    for (uint64_t offset = 0; offset < total_elements; ) {
        // Calculate width of next batch to process (in field elements)
        uint64_t width = total_elements - offset;
        if (width >= AFFT_BATCH_SIZE) {
            width = AFFT_BATCH_SIZE;
        } else if (width % GF16_VT_MIN_BATCH_SIZE != 0) {
            // Round up to batch size. We laready checked that pagesize is a
            // multiple of batch size, so this will not cause invalid reads.
            width += GF16_VT_MIN_BATCH_SIZE - width % GF16_VT_MIN_BATCH_SIZE;
        }

        // Copy input.
        for (int i = 0; i < arg_input_blocks; ++i) {
            const struct InputSlice *s = &input_blocks[i];
            input_rows[s->constant] = s->elems + offset;
        }

        // Batch multiply!
        gf16_vt_batch_mul(output_rows, input_rows, width);

        // Copy output.
        for (int i = 0; i < arg_output_blocks; ++i) {
            struct RecoverySlicePacketBody *body = &recovery_blocks[i]->body;
            memcpy(body->elems + offset, output_rows[i], width * sizeof(gf16_t));
        }

        offset += width;
    }
}

static int compare_input_files_by_id(const void *a, const void *b) {
    return compare_md5(
        &((const struct InputFile *)a)->id,
        &((const struct InputFile *)b)->id);
}

static void sort_input_files_by_id() {
    qsort(input_files, input_file_count, sizeof(struct InputFile),
            compare_input_files_by_id);
}

static void init_input_constants() {
    // From the Par2 spec: input constants are x^n where n is not divisible
    // by 3, 5, 17 or 257.
    int j = 0;
    uint32_t elem = 1;
    for (int n = 0; n < (1 << 16); ++n) {
        if (n%3 != 0 && n%5 != 0 && n%17 != 0 && n%257 != 0) {
            input_blocks[j++].constant = elem;
        }
        elem <<= 1;
		if (elem & 65536) elem ^= GF16_POLY;
    }
    assert(j == MAX_BLOCK_COUNT);
}

static int is_prefix(const char *prefix, const char *full) {
    while (*prefix) {
        int diff = *prefix++ - *full++;
        if (diff != 0) return diff;
    }
    return 0;
}

static uint64_t calculate_max_file_size() {
    uint64_t size = 0;
    for (int i = 0; i < input_file_count; ++i) {
        if (input_files[i].size > size) size = input_files[i].size;
    }
    return size;
}
static uint64_t count_nonempty_files() {
    uint64_t count = 0;
    for (int i = 0; i < input_file_count; ++i) {
        if (input_files[i].size > 0) ++count;
    }
    return count;
}

static uint64_t calculate_block_count(uint64_t block_size) {
    uint64_t count = 0;
    for (int i = 0; i < input_file_count; ++i) {
        count += (input_files[i].size + block_size - 1) / block_size;
    }
    return count;
}

static uint64_t calculate_block_size(uint64_t block_count) {
    assert(block_count >= count_nonempty_files());
    uint64_t lo = 1, hi = 1;
    while (calculate_block_count(4*hi) > block_count) {
        lo = hi + 1;
        hi *= 2;
    }
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo)/2;
        if (calculate_block_count(4*mid) > block_count) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return 4*lo;
}

int parse_arguments(int argc, char *argv[]) {
    if (argc < 2) {
        show_usage();
        return -1;
    }

    if (is_prefix(argv[1], "create") != 0) {
        fprintf(stderr, "Command must be \"create\"\n");
        show_usage();
        return -1;
    }

    // Parse option arguments
    int arg_i = 2;
    while (arg_i < argc && argv[arg_i][0] == '-' && argv[arg_i][1] != '\0') {
        if (parse_option(argv[arg_i]) == 0) {
            ++arg_i;
        } else {
            if (argv[arg_i][2] == '\0' && arg_i + 1 < argc) {
                fprintf(stderr, "Note: no space allowed between option and value. "
                        "Did you mean: \"%s%s\"?\n", argv[arg_i], argv[arg_i + 1]);
            }
            return -1;
        }
    }

    // First positional argument: output file
    if (arg_i == argc) {
        fprintf(stderr, "Missing output filename!\n");
        return -1;
    }
    // Defer opening output until we've opened all the inputs and verified the
    // encoding parameters, to avoid creating an empty file.
    const char *output_filename = argv[arg_i++];

    // Remaining positional arguments: input files
    if (arg_i == argc) {
        fprintf(stderr, "Missing input filenames!\n");
        return -1;
    }
    while (arg_i < argc) {
        if (open_input_file(argv[arg_i++]) != 0) {
            return -1;
        }
    }

    // Fill in default option arguments
    // The logic here is pretty hairy... this probably needs more testing!
    int nonempty_files = (int)count_nonempty_files();
    if (nonempty_files == 0) {
        fprintf(stderr, "No nonempty files in recovery set!\n");
        return -1;
    }
    if (arg_input_blocks == -1) {
        if (arg_block_size == -1) {
            if (nonempty_files > 2000) {
                arg_input_blocks = nonempty_files;
                arg_block_size = calculate_max_file_size();
                if (arg_block_size % 4 != 0) arg_block_size += 4 - arg_block_size % 4;
            } else {
                arg_input_blocks = 2000;
                arg_block_size = calculate_block_size(arg_input_blocks);
            }
        } else if (arg_block_size <= 0 || arg_block_size % 4 != 0) {
            fprintf(stderr, "Block size (%lld) must be a multiple of 4!\n", arg_block_size);
            return -1;
        } else {
            arg_input_blocks = calculate_block_count(arg_block_size);
            if (arg_input_blocks > MAX_BLOCK_COUNT) {
                fprintf(stderr, "Too many input blocks (%lld) for block size!\n", arg_input_blocks);
                return -1;
            }
        }
    } else if (arg_block_size != -1) {
        fprintf(stderr, "Cannot set both block size (-s) and block count (-b)\n");
        return -1;
    } else if (arg_input_blocks < 1 || arg_input_blocks > MAX_BLOCK_COUNT) {
        fprintf(stderr, "Invalid input block count: %lld\n", arg_input_blocks);
        return -1;
    } else if (arg_input_blocks < nonempty_files) {
        fprintf(stderr, "Too few blocks (%lld) for number of nonempty files (%d)\n",
                arg_input_blocks, nonempty_files);
        return -1;
    } else {
        arg_block_size = calculate_block_size(arg_input_blocks);
    }

    int64_t calculated_block_count = calculate_block_count(arg_block_size);
    if (calculated_block_count != arg_input_blocks) {
        fprintf(stderr,
                "Note: calculated block count (%lld) differs from requested block count (%lld).\n"
                "This can happen due to rounding when calculating the block size.\n",
                (int64_t) calculated_block_count, arg_input_blocks);
        arg_input_blocks = calculated_block_count;
    }
    if (calculated_block_count > MAX_BLOCK_COUNT) {
        fprintf(stderr, "Calculated block count (%lld) too high!\n", calculated_block_count);
        return -1;
    }
    if (arg_output_blocks == -1) {
        if (arg_redundancy == -1) {
            arg_redundancy = 5;
        } else if (arg_redundancy < 0 || arg_redundancy > 100) {
            fprintf(stderr, "Invalid redudancy percentage: %lld\n", arg_redundancy);
            return -1;
        }
        arg_output_blocks = (calculated_block_count * arg_redundancy + 50) / 100;
        // Ensure at least 1 output block when redundancy > 0%; this is what other PAR tools do.
        if (arg_redundancy > 0 && arg_output_blocks == 0) arg_output_blocks = 1;
    } else if (arg_redundancy != -1) {
        fprintf(stderr, "Cannot set both redundancy (-r) and recovery block count (-c)\n");
        return -1;
    } else if (arg_output_blocks < 0 || arg_output_blocks > MAX_BLOCK_COUNT) {
        fprintf(stderr, "Invalid recovery block count: %lld\n", arg_input_blocks);
        return -1;
    }
    if (arg_output_start < 0 || arg_output_start > MAX_BLOCK_COUNT) {
        fprintf(stderr, "Invalid recovery block start: %lld (must be between 0 "
            "and %lld when using %lld recovery blocks)\n",
            arg_output_start, MAX_BLOCK_COUNT - arg_output_blocks, arg_output_blocks);
        return -1;
    }

    // Finally create output file.
    if (open_output_file(output_filename) != 0) return -1;

    return 0;
}

#ifdef _MSC_VER // Windows
#include <windows.h>

#define _SC_PAGESIZE 1

long sysconf(int name) {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (name == _SC_PAGESIZE)
        return sysInfo.dwPageSize;
    return -1;
}
#endif

int init_pagesize() {
    long res = sysconf(_SC_PAGESIZE);
    if (res == -1) {
        perror("sysconf");
        return -1;
    }
    if (res == 0 || (res % GF16_VT_MIN_BATCH_SIZE * sizeof(gf16_t) != 0)) {
        fprintf(stderr, "Invalid page size: %ld\n", res);
        return -1;
    }
    pagesize = res;
    return 0;
}

int main(int argc, char *argv[]) {
    int exit_status = 0;

    if (init_pagesize() != 0) goto fail;

    if (parse_arguments(argc, argv) != 0) goto fail;

    if (mmap_input_files() != 0) {
        fprintf(stderr, "Failed to mmap input files.\n");
        goto fail;
    }

    calculate_input_file_ids();
    sort_input_files_by_id();

    // Print a brief summary of what we're about to do.
    fprintf(stderr, "Data blocks:     %10lld\n", arg_input_blocks);
    fprintf(stderr, "Recovery blocks: %10lld\n", arg_output_blocks);
    fprintf(stderr, "Block size:      %10lld (%.1f %s)\n", arg_block_size,
            ((double) arg_block_size / (arg_block_size < (1<<20) ? (1<<10) : (1<<20))),
            (arg_block_size < (1<<20) ? "KiB" : "MiB"));
    fprintf(stderr, "Redundancy:      %10.3f%%\n",
        arg_input_blocks > 0 ? 100.0 * arg_output_blocks / arg_input_blocks : 0);

    // Create main packet. Must be done first to calculate the recovery set id.
    create_main_packet();

    if (write_creator_packet() != 0) {
        fprintf(stderr, "Failed to write creator packet\n");
        goto fail;
    }
    if (write_main_packet() != 0) {
        fprintf(stderr, "Failed to write main packet\n");
        goto fail;
    }
    for (int i = 0; i < input_file_count; ++i) {
        if (write_file_description_packet(&input_files[i]) != 0) {
            fprintf(stderr, "Failed to write file descriptor packet\n");
            goto fail;
        }
    }
    struct InputSlice *input_block_ptr = input_blocks;
    for (int i = 0; i < input_file_count; ++i) {
        if (input_files[i].size == 0) continue;
        if (write_ifsc_packet(&input_files[i], &input_block_ptr) != 0) {
            fprintf(stderr, "Failed to write input slice checksum packet\n");
            goto fail;
        }
    }
    assert(input_block_ptr - input_blocks == arg_input_blocks);

    if (arg_output_blocks == 0) goto finish;

    // Now the fun begins! First, initialize the tables we need:
    gf16_init();
    init_input_constants();
    gf16_vt_mul_init();
    gf16_vt_batch_mul_init();

    if (reserve_output_slices((int)arg_output_blocks) != 0) goto fail;

    // Generate the actual recovery blocks. This is where most time is spent.
    clock_t time_encode = clock();
    if (arg_batch) {
        generate_recovery_data_batch();
    } else {
        generate_recovery_data();
    }
    time_encode = clock() - time_encode;
    if (finalize_output_slices((int)arg_output_blocks) != 0) goto fail;

    // Required time to encode (This doesn't include file access time.)
    double elapsed_secs = (double) time_encode / CLOCKS_PER_SEC;
    double throughput = (double)(arg_block_size * arg_input_blocks) / (1 << 20) / elapsed_secs;
    printf("Time elapsed: %.6f s\n", elapsed_secs);
    printf("Throughput: %.3f MiB/s\n", throughput); // MiB = Mebibyte = 2**20

    exit_status = 0;
    goto finish;

fail:
    exit_status = 1;

finish:

    // TODO: close open files, free memory, etc. (technically not necessary)
    return exit_status;
}
