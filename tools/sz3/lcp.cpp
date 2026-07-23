#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include <iostream>
#include "SZ3/api/sz.hpp"
#include "SZ3/compressor/SZDiscreteCompressor.hpp"
#include "SZ3/frontend/SZGeneralFrontend.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/def.hpp"
#include "SZ3/utils/Statistic.hpp"

void usage() {

    printf("Usage: lcp [options]\n");
    printf("  -i <x> <y> <z>        Input data files, the particle positions of three dimensions to be compressed (required for compression)\n");
    printf("  -z <compressed>       Output data / Input data files for the compressed data (required for compression/decompression)\n");
    printf("  -o <x> <y> <z>        Output data files, the particle positions of three dimensions to be decompressed (may required for decompression)\n");
    printf("  -osn                  The flag that determines the decompressed data files using input names with '.lcp.out' suffix (may required for decompression, -i required for the file names)\n");
    printf("  -eb <value>           Absolute error bound (default 1e-3)\n");
    printf("  -1 <n>                Process a single timestep with n particle points\n");
    printf("  -2 <nt> <n>           Process nt timesteps each with n particle points\n");
    printf("  -bt <block>           Temporal prediction batch size (default 16)\n");
    printf("  -be <mode> [bx by bz] [DO NOT USE] Block size estimation method, do not use this if you don't know\n");
    printf("  -fflag <value>        [DO NOT USE] Precision control for anchor frames, do not use this if you don't know\n");
    printf("  -a                    Keep original data in memory for verification (-i required)\n");
    printf("  -ord <32|64> <file>   Output permutation order using 32- or 64-bit words\n");
    printf("                         Orders are global unless --blockwise-ord is also specified\n");
    printf("  --blockwise-ord       Use the bit-packed block-local order format (single-frame only)\n");
    printf("  --decompress-with-order <32|64> <file>\n");
    printf("                         Apply an order file before writing decompressed particle coordinates\n");
    printf("\n");
    printf("Typical workflows:\n");
    printf("  Compress:   lcp -i x.dat y.dat z.dat -1 <n> -eb <err> -z data.lcp\n");
    printf("  Decompress: lcp -o x.out y.out z.out -1 <n> -z data.lcp\n");
    printf("  Compress:   lcp -i x.dat y.dat z.dat -2 <nt> <n> -eb <err> -z data.lcp\n");
    printf("  Decompress: lcp -o x.out y.out z.out -2 <nt> <n> -z data.lcp\n");
    printf("\n");

    exit(0);
}

size_t max(size_t a, size_t b) {
    return a > b ? a : b;
}

template<class T>
void sortAndVerify(T *a, T *b, size_t *ord, size_t n, std::string str) {

//    std::set<size_t> st;
//    for(size_t i=0;i<n;i++) st.insert(ord[i]);
//    if(*st.begin() == 0 && *st.rbegin() == n-1 && st.size() == n){
//        printf("is a permutation\n");
//    }
//    else{
//        printf("not a permutation\n");
//        printf("min = %zu, max = %zu\n", *st.begin(), *st.rbegin());
//    }

    T *c = new T[n];

    for (size_t i = 0; i < n; i++) {
        c[ord[i]] = b[i];
    }

//    for(size_t i=0;i<n;i++){
//        if(fabs(a[i] - c[i]) > eb){
//            printf("%zu %.12f %.12f %.12f\n", i, c[i], a[i], fabs(a[i] - c[i]));
//            exit(-1);
//        }
//    }

    printf("statistics of %s\n", str.data());

    double psnr, nrmse, max_diff;
    SZ3::verify<T>(a, c, n, psnr, nrmse, max_diff);

    delete[] c;
}

size_t blockOrderBitWidth(size_t block_size) {
    size_t width = 0;
    for (size_t max_index = block_size > 0 ? block_size - 1 : 0; max_index != 0; max_index >>= 1) {
        ++width;
    }
    return width;
}

size_t blockOrderTotalBits(size_t n, const std::vector<size_t> &block_counts) {
    size_t total_bits = 0;
    size_t particle_count = 0;
    for (size_t block_count: block_counts) {
        const size_t width = blockOrderBitWidth(block_count);
        if (width != 0 && block_count > (std::numeric_limits<size_t>::max() - total_bits) / width) {
            printf("Blockwise order file is too large.\n");
            exit(-1);
        }
        if (block_count > std::numeric_limits<size_t>::max() - particle_count) {
            printf("Blockwise particle count is too large.\n");
            exit(-1);
        }
        total_bits += block_count * width;
        particle_count += block_count;
    }

    if (particle_count != n) {
        printf("Block counts contain %zu particles, expected %zu.\n", particle_count, n);
        exit(-1);
    }
    return total_bits;
}

template<class UInt>
void writeBlockwiseOrderFile(const char *path, const size_t *order, size_t n,
                             const std::vector<size_t> &block_counts) {
    static_assert(std::is_unsigned<UInt>::value, "Order words must be unsigned integers");

    constexpr size_t word_bits = std::numeric_limits<UInt>::digits;
    const size_t total_bits = blockOrderTotalBits(n, block_counts);

    const size_t word_count = total_bits / word_bits + (total_bits % word_bits != 0);
    std::vector<UInt> words(word_count, 0);
    size_t bit_position = 0;
    size_t particle = 0;

    for (size_t block_count: block_counts) {
        const size_t width = blockOrderBitWidth(block_count);
        for (size_t i = 0; i < block_count; ++i, ++particle) {
            const size_t value = order[particle];
            if (value >= block_count) {
                printf("Invalid block-local order %zu for block population %zu.\n", value, block_count);
                exit(-1);
            }

            size_t bits_written = 0;
            while (bits_written < width) {
                const size_t word_index = bit_position / word_bits;
                const size_t word_offset = bit_position % word_bits;
                const size_t chunk_bits = std::min(width - bits_written, word_bits - word_offset);
                const UInt mask = chunk_bits == word_bits
                                  ? std::numeric_limits<UInt>::max()
                                  : (UInt(1) << chunk_bits) - 1;
                const UInt chunk = static_cast<UInt>(value >> bits_written) & mask;
                words[word_index] |= chunk << word_offset;
                bit_position += chunk_bits;
                bits_written += chunk_bits;
            }
        }
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        printf("Unable to open order file %s.\n", path);
        exit(-1);
    }
    if (!words.empty()) {
        output.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(UInt));
    }
    output.close();
    if (!output) {
        printf("Unable to write order file %s.\n", path);
        exit(-1);
    }

    printf("blockwise order file = %s (%zu bits, %zu %zu-bit words)\n",
           path, total_bits, word_count, word_bits);
}

template<class UInt>
std::vector<UInt> readOrderWords(const char *path, size_t word_count) {
    if (word_count > std::numeric_limits<size_t>::max() / sizeof(UInt)) {
        printf("Order file is too large.\n");
        exit(-1);
    }
    const size_t expected_bytes = word_count * sizeof(UInt);

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        printf("Unable to open order file %s.\n", path);
        exit(-1);
    }
    const std::streamoff file_size = input.tellg();
    if (file_size < 0 || static_cast<size_t>(file_size) != expected_bytes) {
        printf("Order file %s has %lld bytes, expected %zu.\n",
               path, static_cast<long long>(file_size), expected_bytes);
        exit(-1);
    }

    std::vector<UInt> words(word_count);
    input.seekg(0, std::ios::beg);
    if (!words.empty()) {
        input.read(reinterpret_cast<char *>(words.data()), expected_bytes);
    }
    if (!input) {
        printf("Unable to read order file %s.\n", path);
        exit(-1);
    }
    return words;
}

template<class UInt>
std::vector<size_t> readBlockwiseOrderFile(const char *path, size_t n,
                                           const std::vector<size_t> &block_counts) {
    static_assert(std::is_unsigned<UInt>::value, "Order words must be unsigned integers");

    constexpr size_t word_bits = std::numeric_limits<UInt>::digits;
    const size_t total_bits = blockOrderTotalBits(n, block_counts);
    const size_t word_count = total_bits / word_bits + (total_bits % word_bits != 0);
    const std::vector<UInt> words = readOrderWords<UInt>(path, word_count);

    std::vector<size_t> order(n);
    size_t bit_position = 0;
    size_t particle = 0;
    for (size_t block_count: block_counts) {
        const size_t width = blockOrderBitWidth(block_count);
        std::vector<uchar> seen(block_count, 0);
        for (size_t i = 0; i < block_count; ++i, ++particle) {
            size_t value = 0;
            size_t bits_read = 0;
            while (bits_read < width) {
                const size_t word_index = bit_position / word_bits;
                const size_t word_offset = bit_position % word_bits;
                const size_t chunk_bits = std::min(width - bits_read, word_bits - word_offset);
                const UInt mask = chunk_bits == word_bits
                                  ? std::numeric_limits<UInt>::max()
                                  : (UInt(1) << chunk_bits) - 1;
                const UInt chunk = (words[word_index] >> word_offset) & mask;
                value |= static_cast<size_t>(chunk) << bits_read;
                bit_position += chunk_bits;
                bits_read += chunk_bits;
            }

            if (value >= block_count || seen[value]) {
                printf("Order file %s does not contain a permutation for block population %zu.\n",
                       path, block_count);
                exit(-1);
            }
            seen[value] = 1;
            order[particle] = value;
        }
    }

    printf("blockwise order file applied = %s\n", path);
    return order;
}

template<class UInt>
std::vector<size_t> readGlobalOrderFile(const char *path, size_t n) {
    const std::vector<UInt> words = readOrderWords<UInt>(path, n);
    std::vector<size_t> order(n);
    std::vector<uchar> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const size_t value = static_cast<size_t>(words[i]);
        if (value >= n || seen[value]) {
            printf("Order file %s is not a permutation of 0 through %zu.\n", path, n - 1);
            exit(-1);
        }
        seen[value] = 1;
        order[i] = value;
    }

    printf("global order file applied = %s\n", path);
    return order;
}

std::vector<size_t> readBlockwiseOrderFile(const char *path, size_t word_bits, size_t n,
                                           const std::vector<size_t> &block_counts) {
    if (word_bits == 32) {
        return readBlockwiseOrderFile<uint32_t>(path, n, block_counts);
    }
    if (word_bits == 64) {
        return readBlockwiseOrderFile<uint64_t>(path, n, block_counts);
    }
    printf("Order word size must be 32 or 64.\n");
    exit(-1);
}

std::vector<size_t> readGlobalOrderFile(const char *path, size_t word_bits, size_t n) {
    if (word_bits == 32) {
        return readGlobalOrderFile<uint32_t>(path, n);
    }
    if (word_bits == 64) {
        return readGlobalOrderFile<uint64_t>(path, n);
    }
    printf("Order word size must be 32 or 64.\n");
    exit(-1);
}

std::vector<size_t> getBlockwiseDestinations(const std::vector<size_t> &local_order,
                                             const std::vector<size_t> &block_counts) {
    std::vector<size_t> destinations(local_order.size());
    size_t block_start = 0;
    for (size_t block_count: block_counts) {
        for (size_t i = 0; i < block_count; ++i) {
            destinations[block_start + i] = block_start + local_order[block_start + i];
        }
        block_start += block_count;
    }
    return destinations;
}

template<class T>
void applyOrder(T *datax, T *datay, T *dataz, size_t n, const std::vector<size_t> &destinations) {
    std::vector<T> reordered(n);
    T *data[3] = {datax, datay, dataz};
    for (T *dimension: data) {
        for (size_t i = 0; i < n; ++i) {
            reordered[destinations[i]] = dimension[i];
        }
        memcpy(dimension, reordered.data(), n * sizeof(T));
    }
}

template<class T>
uchar *
compressWithoutAllocateMemory(T *data, SZ3::Config &conf, size_t &outSize, size_t *ord = nullptr, uchar blkflag = 0x00,
                              size_t bx = 0, size_t by = 0, size_t bz = 0) {

    /*
     * the length of data is 3 * conf.num
     */

    const size_t &n = conf.num;

    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compress(conf, data, outSize, ord, blkflag, bx, by, bz);

    return bytes;
}

template<class T>
uchar *
compressWithoutAllocateMemory(T *datax, T *datay, T *dataz, SZ3::Config &conf, size_t &outSize, size_t *ord = nullptr,
                              uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0,
                              size_t *blockwise_ord = nullptr, std::vector<size_t> *block_counts = nullptr) {

    size_t &n = conf.num;

    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compress(conf, datax, datay, dataz, outSize, ord, blkflag, bx, by, bz,
                                       blockwise_ord, block_counts);

    return bytes;
}

/*
 * blkflagfore use this function, make sure you have at least least 12 * sizeof(T) allocated.
 */

template<class T>
uchar *compressWithAllocatedMemory(T *data, SZ3::Config conf, size_t &outSize, uchar *&dest, size_t *ord = nullptr,
                                   uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    uchar *bytes = compressWithoutAllocateMemory(data, conf, outSize, ord, blkflag, bx, by, bz);

    memcpy(dest, bytes, outSize);

    delete[] bytes;

    dest += outSize;

    return dest;
}

template<class T>
uchar *compressWithAllocatedMemory(T *datax, T *datay, T *dataz, SZ3::Config conf, size_t &outSize, uchar *&dest,
                                   size_t *ord = nullptr, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0,
                                   size_t bz = 0) {

    uchar *bytes = compressWithoutAllocateMemory(datax, datay, dataz, conf, outSize, ord, blkflag, bx, by, bz);

    memcpy(dest, bytes, outSize);

    delete[] bytes;

    dest += outSize;

    return dest;
}

template<class T>
void decompressWithoutAllocatedMemory(const uchar *cmpData, T *&datax, T *&datay, T *&dataz, size_t &outSize,
                                      size_t cmpSize, std::vector<size_t> *block_counts = nullptr) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressWithoutAllocateMemory(cmpData, datax, datay, dataz, outSize, cmpSize, block_counts);
}

template<class T>
void
decompressWithAllocatedMemory(const uchar *cmpData, T *datax, T *datay, T *dataz, size_t &outSize, size_t cmpSize) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressWithAllocateMemory(cmpData, datax, datay, dataz, outSize, cmpSize);
}

template<class T>
void
decompressWithAllocatedMemoryWithTemporalPrediction(const uchar *cmpData, T *datax, T *datay, T *dataz, size_t &outSize,
                                                    size_t cmpSize) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressSimpleBlockingWithTemporalPrediction(cmpData, datax, datay, dataz, outSize, cmpSize);
}

template<class T>
void deleteData(T *datax, T *datay, T *dataz, size_t n) {

    if (datay + n != dataz) delete[] dataz;
    if (datax + n != datay) delete[] datay;
    delete[] datax;
}

template<class T>
T *readFile(char *inPath[], size_t num_inPath, size_t n) {

    T *data = new T[n * num_inPath];
    for (size_t i = 0; i < num_inPath; i++) {
        SZ3::readfile<T>(inPath[i], n, data + i * n);
    }

    return data;
}

template<class T>
void compress1(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata = nullptr,
               size_t *ord = nullptr, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0,
               size_t *blockwise_ord = nullptr, std::vector<size_t> *block_counts = nullptr) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t outSize = 0;

    SZ3::Timer timer(true);

    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;

    uchar *bytes = compressWithoutAllocateMemory<T>(datax, datay, dataz, conf, outSize, ord, blkflag, bx, by, bz,
                                                    blockwise_ord, block_counts);
    delete[] data;

    double compress_time = timer.stop();

    SZ3::writefile(cmpPath, bytes, outSize);
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

template<class T>
void compress2(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata, size_t *ord,
               uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);
    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;
    T *tailx = datax, *taily = datay, *tailz = dataz;
    size_t *ordt = ord;

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t &nt = conf.dims[0], &n = conf.dims[1];

    uchar *bytes = new uchar[max(conf.num * 12, 65536)], *tail = bytes + (nt + 2) * sizeof(size_t), *tailp = bytes;
    SZ3::write(nt, tailp);
    SZ3::write(n, tailp);

    size_t outSize = 0, sumOutSize = 0;
    SZ3::Config confSlice(n);
    confSlice.absErrorBound = conf.absErrorBound;

    SZ3::Timer timer(true);

    if (ord == nullptr) {

        for (size_t t = 0; t < nt; t++) {

//            printf("t = %zu\n", t);

            compressWithAllocatedMemory(tailx, taily, tailz, confSlice, outSize, tail, nullptr, blkflag, bx, by, bz);

//            printf("t = %zu outSize = %zu\n", t, outSize);

            sumOutSize += outSize;
            SZ3::write(outSize, tailp);

            tailx += n;
            taily += n;
            tailz += n;
        }
    } else {
        for (size_t t = 0; t < nt; t++) {

//            printf("t = %zu\n", t);

            compressWithAllocatedMemory(tailx, taily, tailz, confSlice, outSize, tail, ordt, blkflag, bx, by, bz);

//            printf("t = %zu outSize = %zu\n", t, outSize);

            for (size_t i = 0; i < n; i++) ord[t * n + i] += t * n;

            sumOutSize += outSize;
            SZ3::write(outSize, tailp);

            tailx += n;
            taily += n;
            tailz += n;
            ordt += n;
        }
    }


    double compress_time = timer.stop();

    deleteData(datax, datay, dataz, conf.num);

    SZ3::writefile(cmpPath, bytes, (size_t) (tail - bytes));
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / (size_t) (tail - bytes));
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

template<class T>
void compress2WithTemporalPrediction(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata,
                                     size_t *ord, size_t bt, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0,
                                     size_t bz = 0, double fflagInit = 0) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);
    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;
    T *tailx = datax, *taily = datay, *tailz = dataz;
    size_t *ordt = ord;

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t &nt = conf.dims[0], &n = conf.dims[1];

    size_t outSize = 0;

    SZ3::Timer timer(true);

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compressSimpleBlockingWithTemporalPrediction(conf, bt, datax, datay, dataz, outSize, ord,
                                                                           blkflag, bx, by, bz, fflagInit);

    double compress_time = timer.stop();

    deleteData(datax, datay, dataz, conf.num);

    SZ3::writefile(cmpPath, bytes, outSize);
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

#include <random>

template<class T>
void shuffle3(T *&data, size_t n) {

    SZ3::Timer timer(true);

//    std::mt19937 rng(std::random_device{}());
    std::mt19937 rng(16707);

    size_t *p = new size_t[n];
    for (int i = 0; i < n; i++) p[i] = i;
    std::shuffle(p, p + n, rng);

    T *shuffled_data = new T[n * 3];
    for (int i = 0; i < n; i++) shuffled_data[i] = data[p[i]];
    for (int i = 0; i < n; i++) shuffled_data[n + i] = data[n + p[i]];
    for (int i = 0; i < n; i++) shuffled_data[2 * n + i] = data[2 * n + p[i]];

    delete[] p;
    delete[] data;

    data = shuffled_data;

    double shuffle_time = timer.stop();
    printf("shuffle time = %.2lf\n", shuffle_time);
}

//template <class T>
//void blockingCompress(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf){
//
//    SZ3::Timer timer(true);
//
//    T *data = new T[conf.num*3];
//    for(int i=0;i<num_inPath;i++){
//        SZ3::readfile<T>(inPath[i], conf.num, data + i * conf.num);
//    }
//
////    shuffle3(data, conf.num);
//
//    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());
//
//    std::vector<size_t> szvec;
//    size_t n = conf.num;
//    while(n){
//        szvec.push_back(n-n/2);
//        n=n/2;
//    }
//    std::reverse(szvec.begin(), szvec.end());
//
////    for(size_t it:sz){
////        std::cout<<it<<' ';
////    }
////    std::cout<<std::endl;
//
//    uchar *bytes = new uchar[conf.num * 12], *tail = bytes;
//    size_t totalSize = 0, ssz = 0;
//
//    tail += 4 * (szvec.size() + 2);
//    uchar *tailp = bytes;
//    SZ3::write(conf.num, tailp);
//
//    for(size_t bsz:szvec){
//        SZ3::Config bconf = conf;
//        bconf.num = bsz;
////        printf("bsz = %zu\n",bsz);
//        size_t bSize;
//        T *bdata = new T[bsz * 3];
//        memcpy(bdata, data + ssz, bsz * sizeof(T));
//        memcpy(bdata + bsz, data + (conf.num + ssz), bsz * sizeof(T));
//        memcpy(bdata + 2 * bsz, data + (2 * conf.num + ssz), bsz * sizeof(T));
//
//        uchar *bbytes = compressor.compress(bconf, bdata, bSize);
//        memcpy(tail, bbytes, bSize);
//        delete[] bbytes;
//        SZ3::write((size_t)(tail - bytes), tailp);
//        tail += bSize;
//
//        ssz += bsz;
//    }
//
//    SZ3::write((size_t)(tail - bytes), tailp);
//
//    totalSize = tail - bytes;
//
//    SZ3::writefile(cmpPath, bytes, totalSize);
//    delete[] data;
//    delete[] bytes;
//
//    double compress_time = timer.stop();
//
//    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / totalSize);
//    printf("compression time = %f\n", compress_time);
//    printf("compressed data file = %s\n", cmpPath);
//}

template<class T>
void decompress1(char *outPath[], size_t num_outPath, char *cmpPath, T *oridata = nullptr, size_t *ord = nullptr,
                 const char *order_path = nullptr, size_t order_word_bits = 0, bool blockwise_order = false) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailData = cmpData.get();

    T *datax = nullptr, *datay = nullptr, *dataz = nullptr;

    size_t n;
    std::vector<size_t> block_counts;

    SZ3::Timer timer(true);

    decompressWithoutAllocatedMemory(tailData, datax, datay, dataz, n, cmpSize,
                                     order_path != nullptr && blockwise_order ? &block_counts : nullptr);

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, n, "x");
        sortAndVerify(oridata + n, datay, ord, n, "y");
        sortAndVerify(oridata + n + n, dataz, ord, n, "z");
    }

    if (order_path != nullptr) {
        if (blockwise_order) {
            const std::vector<size_t> local_order =
                    readBlockwiseOrderFile(order_path, order_word_bits, n, block_counts);
            const std::vector<size_t> destinations = getBlockwiseDestinations(local_order, block_counts);
            applyOrder(datax, datay, dataz, n, destinations);
        } else {
            const std::vector<size_t> destinations =
                    readGlobalOrderFile(order_path, order_word_bits, n);
            applyOrder(datax, datay, dataz, n, destinations);
        }
    }

    SZ3::writefile(outPath[0], datax, n);
    SZ3::writefile(outPath[1], datay, n);
    SZ3::writefile(outPath[2], dataz, n);

    deleteData(datax, datay, dataz, n);

    printf("decompression time = %f\n", compress_time);
}

template<class T>
void decompress2(char *outPath[], int num_outPath, char *cmpPath, T *oridata = nullptr, size_t *ord = nullptr,
                 const char *order_path = nullptr, size_t order_word_bits = 0) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailPos = cmpData.get();
    size_t nt, n;
    SZ3::read(nt, tailPos);
    SZ3::read(n, tailPos);
    const uchar *tailData = tailPos + nt * sizeof(size_t);

    T *datax = new T[nt * n * 3], *datay = datax + nt * n, *dataz = datay + nt * n;
    T *tailx = datax, *taily = datay, *tailz = dataz;

    SZ3::Timer timer(true);

    for (size_t t = 0; t < nt; t++) {

        size_t n;
        size_t cmpSize;
        SZ3::read(cmpSize, tailPos);

        decompressWithAllocatedMemory(tailData, tailx, taily, tailz, n, cmpSize);
        tailData += cmpSize;
        tailx += n;
        taily += n;
        tailz += n;
    }

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, nt * n, "x");
        sortAndVerify(oridata + nt * n, datay, ord, nt * n, "y");
        sortAndVerify(oridata + 2 * nt * n, dataz, ord, nt * n, "z");
    }

    if (order_path != nullptr) {
        const std::vector<size_t> destinations =
                readGlobalOrderFile(order_path, order_word_bits, nt * n);
        applyOrder(datax, datay, dataz, nt * n, destinations);
    }

    SZ3::writefile(outPath[0], datax, nt * n);
    SZ3::writefile(outPath[1], datay, nt * n);
    SZ3::writefile(outPath[2], dataz, nt * n);

    deleteData(datax, datay, dataz, nt * n);

    printf("decompression time = %f\n", compress_time);
}

template<class T>
void decompress2WithTemporalPrediction(char *outPath[], int num_outPath, char *cmpPath, T *oridata = nullptr,
                                       size_t *ord = nullptr, const char *order_path = nullptr,
                                       size_t order_word_bits = 0) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailPos = cmpData.get();
    size_t nt, n, bt;

    //
    {
        int cnt1;
        SZ3::read(cnt1, tailPos);
        while (cnt1--) {
            size_t tem;
            SZ3::read(tem, tailPos);
            tailPos += tem;
        }
    }

    SZ3::read(nt, tailPos);
    SZ3::read(n, tailPos);
    SZ3::read(bt, tailPos);

    T *datax = new T[nt * n * 3], *datay = datax + nt * n, *dataz = datay + nt * n;

    SZ3::Timer timer(true);

    size_t outSize;

    decompressWithAllocatedMemoryWithTemporalPrediction(cmpData.get(), datax, datay, dataz, outSize, cmpSize);

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, nt * n, "x");
        sortAndVerify(oridata + nt * n, datay, ord, nt * n, "y");
        sortAndVerify(oridata + 2 * nt * n, dataz, ord, nt * n, "z");
    }

    if (order_path != nullptr) {
        const std::vector<size_t> destinations =
                readGlobalOrderFile(order_path, order_word_bits, nt * n);
        applyOrder(datax, datay, dataz, nt * n, destinations);
    }

    SZ3::writefile(outPath[0], datax, nt * n);
    SZ3::writefile(outPath[1], datay, nt * n);
    SZ3::writefile(outPath[2], dataz, nt * n);

    deleteData(datax, datay, dataz, nt * n);

    printf("decompression time = %f\n", compress_time);
}

//template <class T>
//void blockingDecompress(char *outPath[], int num_outPath, char *cmpPath, double ratio){

//    SZ3::Timer timer(true);
//
//    size_t cmpSize;
//    const auto blockingData = SZ3::readfile<uchar>(cmpPath, cmpSize);
//    const uchar *blockingData_tail = blockingData.get();
//    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());
//
//    std::vector<size_t> szvec, posvec;
//    size_t n = 0;
//    SZ3::read(n, blockingData_tail);
//    {
//        size_t tem = n;
//        while (tem) {
//            szvec.push_back(tem - tem / 2);
//            tem = tem / 2;
//        }
//    }
//    std::reverse(szvec.begin(), szvec.end());
//    posvec.resize(szvec.size() + 1);
//    for(size_t &it:posvec){
//        SZ3::read(it, blockingData_tail);
//        printf("%zu\n",it);
//    }
//    size_t ssz = 0;
//    std::vector<size_t> bno;
//    for(int i=szvec.size()-1;i>=0;i--){
//        if(1. * (ssz + szvec[i]) / n <= ratio){
//            bno.push_back(i);
//            ssz += szvec[i];
////            printf("(ssz + szvec[i]) / n = (%zu + %zu) / %zu = %f <= %f\n",ssz,szvec[i],n,1. * (ssz + szvec[i]) / n,ratio);
//        }
////        else{
////            printf("(ssz + szvec[i]) / n = (%zu + %zu) / %zu = %f > %f\n",ssz,szvec[i],n,1. * (ssz + szvec[i]) / n,ratio);
////        }
//    }
//
//    printf("ssz = %zu\n",ssz);
//
//    T *decmpData = new T[ssz*3];
//
//    // need to be done
//
//    size_t p = 0;
//
//    for(size_t it:bno){
//
//        printf("bid = %zu, sz = %zu\n", it, szvec[it]);
//
//
//    }
//
//    for(int i=0;i<num_outPath;i++){
//        SZ3::writefile(outPath[i], decmpData + i * ssz, ssz);
//    }
//
//    delete[] decmpData;
//
//    double compress_time = timer.stop();
//
//    printf("decompression time = %f\n", compress_time);
//}

signed main(int argc, char *argv[]) {

//    static SZ3::LinearQuantizer<float> quantizer(0.1, 65536);
//
//    float a = 141.595322, b = 141.595322;
//    int e = quantizer.quantize(a,b) - 65536;
//
//    printf("%d\n", e);
//    printf("%f %f\n", a, b);
//
//    return 0;

    char **inPath = new char *[3];
    char **outPath = new char *[3];
    for (int i = 0; i < 3; i++) inPath[i] = new char[1024];
    for (int i = 0; i < 3; i++) outPath[i] = new char[1024];
    char cmpPath[1024];
    char ordPath[1024];
    char inputOrdPath[1024];

    uchar cmp = 0x00, decmp = 0x00, flag = 0x00, output_ord = 0x00, input_ord = 0x00;
    bool blockwise_order = false;
    size_t ordBits = 64;
    size_t inputOrdBits = 64;
    /*
     * flag = 1, compress without time dimension
     * flag = 2, compress with time dimension
     * flag = 3, blocking compression without time dimension
     */
    size_t nt, n;
    // default eb = 1e-3
    double eb = 1e-3;
    // default bt = 16
    size_t bt = 16;

    uchar blkflag = 0x03;
    size_t bx = 0, by = 0, bz = 0;

    double fflagInit = 0;

    uchar _a = 0x00;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            assert(i + 3 < argc);
            snprintf(inPath[0], 1024, "%s", argv[i + 1]);
            snprintf(inPath[1], 1024, "%s", argv[i + 2]);
            snprintf(inPath[2], 1024, "%s", argv[i + 3]);
            cmp = 0x01;
            i += 3;
        } else if (strcmp(argv[i], "-z") == 0) {
            assert(i + 1 < argc);
            snprintf(cmpPath, 1024, "%s", argv[i + 1]);
            i += 1;
        } else if (strcmp(argv[i], "-o") == 0) {
            assert(i + 3 < argc);
            snprintf(outPath[0], 1024, "%s", argv[i + 1]);
            snprintf(outPath[1], 1024, "%s", argv[i + 2]);
            snprintf(outPath[2], 1024, "%s", argv[i + 3]);
            decmp = 0x01;
            i += 3;
        } else if (strcmp(argv[i], "-osn") == 0) {
            // output same name
            snprintf(outPath[0], 1024, "%s.lcp.out", inPath[0]);
            snprintf(outPath[1], 1024, "%s.lcp.out", inPath[1]);
            snprintf(outPath[2], 1024, "%s.lcp.out", inPath[2]);
            decmp = 0x01;
        } else if (strcmp(argv[i], "-eb") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%lf", &eb);
            i += 1;
        } else if (strcmp(argv[i], "-1") == 0) {
            assert(i + 1 < argc);
            flag = 1;
            sscanf(argv[i + 1], "%zu", &n);
            i += 1;
        } else if (strcmp(argv[i], "-2") == 0) {
            assert(i + 2 < argc);
            flag = 2;
            sscanf(argv[i + 1], "%zu", &nt);
            sscanf(argv[i + 2], "%zu", &n);
            i += 2;
        } else if (strcmp(argv[i], "-b") == 0) {

        } else if (strcmp(argv[i], "-a") == 0) {
            _a = 0x01;
        } else if (strcmp(argv[i], "-bt") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%zu", &bt);
            i += 1;
        } else if (strcmp(argv[i], "-be") == 0) {
            assert(i + 1 < argc);
            size_t tem;
            sscanf(argv[i + 1], "%zu", &tem);
            blkflag = tem;
            i += 1;
            if (blkflag == 0x00) {
                assert(i + 3 < argc);
                sscanf(argv[i + 1], "%zu", &bx);
                sscanf(argv[i + 2], "%zu", &by);
                sscanf(argv[i + 3], "%zu", &bz);
                i += 3;
            }
        } else if (strcmp(argv[i], "-fflag") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%lf", &fflagInit);
            i += 1;
        } else if (strcmp(argv[i], "-ord") == 0) {
            assert(i + 2 < argc);
            sscanf(argv[i + 1], "%zu", &ordBits);
//            assert(ordBits == 32 || ordBits == 64);
            if (ordBits != 32 && ordBits != 64) {
                printf("ordBits must be 32 or 64.\n");
                exit(-1);
            }
            snprintf(ordPath, 1024, "%s", argv[i + 2]);
            output_ord = 0x01;
            i += 2;
        } else if (strcmp(argv[i], "--blockwise-ord") == 0) {
            blockwise_order = true;
        } else if (strcmp(argv[i], "--decompress-with-order") == 0) {
            assert(i + 2 < argc);
            sscanf(argv[i + 1], "%zu", &inputOrdBits);
            if (inputOrdBits != 32 && inputOrdBits != 64) {
                printf("Order word size must be 32 or 64.\n");
                exit(-1);
            }
            snprintf(inputOrdPath, 1024, "%s", argv[i + 2]);
            input_ord = 0x01;
            i += 2;
        } else {
            usage();
        }
    }

    if (input_ord && !decmp) {
        printf("--decompress-with-order requires -o or -osn.\n");
        exit(-1);
    }
    if (blockwise_order && !output_ord && !input_ord) {
        printf("--blockwise-ord requires -ord or --decompress-with-order.\n");
        exit(-1);
    }
    if (blockwise_order && flag != 1) {
        printf("--blockwise-ord requires single-frame mode (-1).\n");
        exit(-1);
    }

    SZ3::Config conf(n);
    if (flag == 2) conf = SZ3::Config(nt, n);

    conf.absErrorBound = eb;

    float *oridata = nullptr;
    size_t *ord = nullptr;
    std::vector<size_t> blockwise_ord;
    std::vector<size_t> block_counts;

    if (_a || output_ord) {
        if (cmp == 0x00) {
            printf("Must contain input while using -a or -ord\n");
            exit(-1);
        }
        oridata = new float[conf.num * 3];
        ord = new size_t[conf.num];
        if (output_ord && blockwise_order) blockwise_ord.resize(conf.num);
    }

    if (cmp == 0x01) {
        switch (flag) {
            case 1: {
                compress1<float>(inPath, 3, cmpPath, conf, oridata, ord, blkflag, bx, by, bz,
                                 blockwise_ord.empty() ? nullptr : blockwise_ord.data(),
                                 blockwise_ord.empty() ? nullptr : &block_counts);
                break;
            }
            case 2: {
                if (bt > 0) {
                    compress2WithTemporalPrediction<float>(inPath, 3, cmpPath, conf, oridata, ord, bt, blkflag, bx, by,
                                                           bz, fflagInit);
                } else {
                    compress2<float>(inPath, 3, cmpPath, conf, oridata, ord, blkflag, bx, by, bz);
                }

                break;
            }
            case 3: {
//                blockingCompress<float>(inPath, 3, cmpPath, conf);
                break;
            }
            default: {
                exit(0);
            }
        }
    }

    if (output_ord) {
        if (blockwise_order) {
            if (ordBits == 32) {
                writeBlockwiseOrderFile<uint32_t>(ordPath, blockwise_ord.data(), conf.num, block_counts);
            } else {
                writeBlockwiseOrderFile<uint64_t>(ordPath, blockwise_ord.data(), conf.num, block_counts);
            }
        } else if (ordBits == 32) {
            uint32_t *ordu32 = new uint32_t[conf.num];
            for (size_t i = 0; i < conf.num; i++) {
                ordu32[i] = ord[i];
            }
            SZ3::writefile(ordPath, ordu32, conf.num);
            delete[] ordu32;
        }
        else {
            uint64_t *ordu64 = new uint64_t[conf.num];
            for (size_t i = 0; i < conf.num; i++) {
                ordu64[i] = ord[i];
            }
            SZ3::writefile(ordPath, ordu64, conf.num);
            delete[] ordu64;
        }
    }

    if (decmp == 0x01) {
        switch (flag) {
            case 1: {
                decompress1<float>(outPath, 3, cmpPath, oridata, ord,
                                   input_ord ? inputOrdPath : nullptr, inputOrdBits, blockwise_order);
                break;
            }
            case 2: {
                if (bt > 0) {
                    decompress2WithTemporalPrediction<float>(outPath, 3, cmpPath, oridata, ord,
                                                              input_ord ? inputOrdPath : nullptr, inputOrdBits);
                } else {
                    decompress2<float>(outPath, 3, cmpPath, oridata, ord,
                                       input_ord ? inputOrdPath : nullptr, inputOrdBits);
                }
                break;
            }
            case 3: {
//                blockingDecompress<float>(outPath, 3, cmpPath, 0.8);
                break;
            }
            default: {
                exit(0);
            }
        }
    }

    return 0;
}
