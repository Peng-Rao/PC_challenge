#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

/// @brief  check if array is sorted correctly
/// @param ref  reference array
/// @param data  data array
/// @param size  size of the arrays
/// @return  true if data is sorted correctly
bool isSorted(int ref[], const int data[], const size_t size) {
    std::sort(ref, ref + size);
    for (size_t idx = 0; idx < size; ++idx) {
        if (ref[idx] != data[idx]) {
            return false;
        }
    }
    return true;
}

/// @brief  sequential merge step (straight-forward implementation)
/// @param out  output array
/// @param in  input array
/// @param begin1  begin of the first array
/// @param end1  end of the first array
/// @param begin2  begin of the second array
/// @param end2  end of the second array
/// @param outBegin  begin of the output array
void MsMergeSequential(int *out, const int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
    long left = begin1;
    long right = begin2;
    long idx = outBegin;

    while (left < end1 && right < end2) {
        if (in[left] <= in[right]) {
            out[idx] = in[left];
            left++;
        } else {
            out[idx] = in[right];
            right++;
        }
        idx++;
    }

    while (left < end1) {
        out[idx] = in[left];
        left++, idx++;
    }

    while (right < end2) {
        out[idx] = in[right];
        right++, idx++;
    }
}

/// @brief sequential merge sort
/// @param array  input array
/// @param tmp  temporary array
/// @param inplace  flag to indicate if the result should be stored in the input array
/// @param begin  begin of the array
/// @param end  end of the array
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
    if (begin < (end - 1)) {
        const long half = (begin + end) / 2;
        MsSequential(array, tmp, !inplace, begin, half);
        MsSequential(array, tmp, !inplace, half, end);
        if (inplace) {
            MsMergeSequential(array, tmp, begin, half, half, end, begin);
        } else {
            MsMergeSequential(tmp, array, begin, half, half, end, begin);
        }
    } else if (!inplace) {
        tmp[begin] = array[begin];
    }
}

/// @brief Parallel MergeSort
/// @param array  input array
/// @param tmp  temporary array
/// @param inplace  flag to indicate if the result should be stored in the input array
/// @param begin  begin of the array
/// @param end  end of the array
/// @param deep  recursion depth
void MsParallel(int *array, int *tmp, bool inplace, long begin, long end, int deep) {
    if (begin < (end - 1)) {
        long half = (begin + end) / 2;
        if (deep) {
            #pragma omp task default(none) \
                        shared(array, tmp) \
                        shared(half, begin, end) \
                        shared(deep) \
                        firstprivate(inplace)
            {
                MsParallel(array, tmp, !inplace, begin, half, deep - 1);
            }
            #pragma omp task default(none) \
                shared(array, tmp) \
                shared(half, begin, end) \
                shared(deep) \
                firstprivate(inplace)
            {
                MsParallel(array, tmp, !inplace, half, end, deep - 1);
            }
            #pragma omp taskwait
        } else {
            MsSequential(array, tmp, !inplace, begin, half);
            MsSequential(array, tmp, !inplace, half, end);
        }

        if (inplace) {
            MsMergeSequential(array, tmp, begin, half, half, end, begin);
        } else {
            MsMergeSequential(tmp, array, begin, half, half, end, begin);
        }
    } else if (!inplace) {
        tmp[begin] = array[begin];
    }
}

/// @brief  serial merge sort
/// @param array  input array
/// @param tmp  temporary array
/// @param size  size of the array
void MsSerial(int *array, int *tmp, const size_t size) {
    MsSequential(array, tmp, true, 0, size);
}


int main(int argc, char *argv[]) {
    // variables to measure the elapsed time
    timeval t1{}, t2{};
    double etime;

    // expect one command line argument: array size
    if (argc != 2) {
        printf("Usage: MergeSort.exe <array size> \n");
        printf("\n");
        return EXIT_FAILURE;
    }
    const size_t stSize = std::strtol(argv[1], nullptr, 10);
    std::cout << "Array size: " << stSize << std::endl;
    int *data = (int *) malloc(stSize * sizeof(int));
    int *tmp = (int *) malloc(stSize * sizeof(int));
    int *ref = (int *) malloc(stSize * sizeof(int));

    printf("Initialization...\n");

    std::srand(95);
    for (size_t idx = 0; idx < stSize; ++idx) {
        data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
    }
    std::copy(data, data + stSize, ref);

    const double dSize = stSize * sizeof(int) / 1024 / 1024;
    printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

    // sequential sorting
    gettimeofday(&t1, nullptr);
    MsSerial(data, tmp, stSize);
    gettimeofday(&t2, nullptr);
    etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    etime = etime / 1000;
    printf("Serial sorting done, took %f sec. Verification...", etime);
    if (isSorted(ref, data, stSize)) {
        printf(" successful.\n");
    } else {
        printf(" FAILED.\n");
    }

    free(data);
    free(tmp);
    free(ref);

    return EXIT_SUCCESS;
}