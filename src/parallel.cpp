#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>

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

///
/// @param out  output array
/// @param in   input array
/// @param begin1  begin of the first array
/// @param end1  end of the first array
/// @param begin2  begin of the second array
/// @param end2  end of the second array
/// @param outBegin  begin of the output array
/// @param deep  recursion depth
void MsMergeParallel(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, int deep) {
    // add cutoff for small arrays
    if (end1 - begin1 + end2 - begin2 < 1000) {
        MsMergeSequential(out, in, begin1, end1, begin2, end2, outBegin);
        return;
    }

    if (deep) {
        long half1, half2, tmp, count, step;
        if (end1 - begin1 < end2 - begin2) {
            half2 = (begin2 + end2) / 2;
            // find in[half2] in [begin1, end1) (std::upper_bound)
            half1 = begin1, count = (end1 - begin1);
            while (count > 0) {
                step = count / 2;
                tmp = half1 + step;
                if (in[tmp] <= in[half2]) {
                    tmp++;
                    half1 = tmp;
                    count -= step + 1;
                } else {
                    count = step;
                }
            }
        } else {
            half1 = (begin1 + end1) / 2;
            // find in[half1] in [begin2, end2) (std::lower_bound)
            half2 = begin2, count = (end2 - begin2);
            while (count > 0) {
                step = count / 2;
                tmp = half2 + step;
                if (in[tmp] < in[half1]) {
                    tmp++;
                    half2 = tmp;
                    count -= step + 1;
                } else {
                    count = step;
                }
            }
        }

        #pragma omp task default(none) \
                    shared(out, in) \
                    shared(begin1, half1, begin2, half2) \
                    shared(outBegin, deep)
        {
            MsMergeParallel(out, in, begin1, half1, begin2, half2, outBegin, deep - 1);
        }

        long outBegin2 = outBegin + (half1 - begin1) + (half2 - begin2);
        #pragma omp task default(none) \
                    shared(out, in) \
                    shared(half1, end1, half2, end2) \
                    shared(outBegin2, deep)
        {
            MsMergeParallel(out, in, half1, end1, half2, end2, outBegin2, deep - 1);
        }
        #pragma omp taskwait
    } else {
        MsMergeSequential(out, in, begin1, end1, begin2, end2, outBegin);
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
    if (begin < end - 1) {
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
            MsMergeParallel(array, tmp, begin, half, half, end, begin, deep);
        } else {
            MsMergeParallel(tmp, array, begin, half, half, end, begin, deep);
        }
    } else if (!inplace) {
        tmp[begin] = array[begin];
    }
}

/// @brief  parallel merge sort with OpenMP
/// @param array input array
/// @param tmp temporary array
/// @param size size of the array
/// @param depth recursion depth
void MsOpenMP(int *array, int *tmp, const size_t size, const int depth) {
#pragma omp parallel default(none) \
shared(array, tmp, size, depth)
#pragma omp single
    {
        MsParallel(array, tmp, true, 0, size, depth);
    }
}

/// @brief  structure to store the test results
/// @param array_size  size of the array
/// @param depth  recursion depth
/// @param time  elapsed time
/// @param success  flag to indicate if the test was successful
struct TestResult {
    size_t array_size;
    int depth;
    double time;
    bool success;
};

/// @brief  run a single test
/// @param size  size of the array
/// @param depth  recursion depth
/// @return  test result
TestResult runSingleTest(size_t size, int depth) {
    TestResult result{};
    result.array_size = size;
    result.depth = depth;

    int *data = (int *) malloc(size * sizeof(int));
    int *tmp = (int *) malloc(size * sizeof(int));
    int *ref = (int *) malloc(size * sizeof(int));

    std::srand(95); // NOLINT(*-msc51-cpp)
    for (size_t idx = 0; idx < size; ++idx) {
        data[idx] = (int) (size * (double(rand()) / RAND_MAX));
    }
    std::copy(data, data + size, ref);

    timeval t1{}, t2{};
    gettimeofday(&t1, nullptr);
    MsOpenMP(data, tmp, size, depth);
    gettimeofday(&t2, nullptr);

    result.time = ((t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000.0;
    result.success = isSorted(ref, data, size);

    free(data);
    free(tmp);
    free(ref);

    return result;
}


/// @brief  write the test results to a CSV file
/// @param results  test results
/// @param filename  name of the file
void writeResultsToCSV(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "Array Size,Depth,Time (seconds),Success\n";

    for (const auto& result : results) {
        file << result.array_size << ","
             << result.depth << ","
             << result.time << ","
             << (result.success ? "true" : "false") << "\n";
    }

    file.close();
}


int main() {
    std::vector<TestResult> all_results;
    std::vector<size_t> sizes;
    const std::vector depths = {5, 8, 10, 12, 15};

    for (int i = 2; i <= 8; ++i) {
        sizes.push_back(static_cast<size_t>(std::pow(10, i)));
    }

    printf("Starting benchmark tests...\n");

    for (const size_t size : sizes) {
        printf("Testing array size: %zu\n", size);
        double dSize = (size * sizeof(int)) / 1024 / 1024;
        printf("  Size: %.2f MB\n", dSize);
        for (const int depth : depths) {
            printf("  With depth: %d\n", depth);
            TestResult result = runSingleTest(size, depth);
            all_results.push_back(result);

            printf("    Time: %.3f seconds, Success: %s\n",
                   result.time,
                   result.success ? "true" : "false");
        }
    }

    writeResultsToCSV(all_results, "merge_sort_parallel_results.csv");
    printf("\nResults have been written to merge_sort_parallel_results.csv\n");
    return EXIT_SUCCESS;
}