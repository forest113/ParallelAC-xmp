# CMakeLists.txt for parallel alpha complex project                                                                                                                                                                          
project(parallel-alpha-complex)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)

cuda_compile(PARALLEL_AC_O AlphaComplexParallel.cu)
cuda_add_executable(parallelac ${PARALLEL_AC_O})

cuda_compile(PARALLEL_AC_LD_O AlphaComplexLargeData.cu)
cuda_add_executable(parallelac-largeData ${PARALLEL_AC_LD_O})
