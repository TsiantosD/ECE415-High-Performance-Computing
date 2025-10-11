# Code Optimizations
The optimizations made to the code are noted here. Each version has all the previous optimizations plus the new ones. All runs were made on the same performance core of a 12th Gen Intel® Core™ i5-1235U × 12 CPU without any compiler optimizations (`-O0` option).

### 1) Original algorithm
- Golden version 🪙️
- Sloooow 🐌️

| Executable                     |  #Runs |  Average (s) |  Std Dev (s) |
| ------------------------------ | ------ | ------------ | ------------ |
| 1_sobel_orig                   |     10 |      1.71490 |      0.03946 |


### 2) Loop interchange
- Switch the loops from column first to row first for better cache locality on the following loops:
  - The `convolution2D` function loop
  - The `sobel` function main loop that applies the filter
  - The `sobel` function loop that calculates the PSNR

| Executable                     |  #Runs |  Average (s) |  Std Dev (s) |
| ------------------------------ | ------ | ------------ | ------------ |
| 1_sobel_orig                   |     10 |      1.71490 |      0.03946 |
| 2_sobel_loop_interchange       |     10 |      1.18228 |      0.00969 |


### 3) Loop unrolling
- Fully unroll the `convolution2D` function loop
- Unroll the main main loop and the PSNR calculation loop with a factor of `4`, handle left over columns with a `switch` statement

| Executable                     |  #Runs |  Average (s) |  Std Dev (s) |
| ------------------------------ | ------ | ------------ | ------------ |
| 1_sobel_orig                   |     10 |      1.71490 |      0.03946 |
| 2_sobel_loop_interchange       |     10 |      1.18228 |      0.00969 |
| 3_sobel_loop_unrolling         |     10 |      0.89895 |      0.00957 |


### 4) Loop fusion
- Fuse together the PSNR calculation and the sobel operator

| Executable                     |  #Runs |  Average (s) |  Std Dev (s) |
| ------------------------------ | ------ | ------------ | ------------ |
| 1_sobel_orig                   |     10 |      1.71490 |      0.03946 |
| 2_sobel_loop_interchange       |     10 |      1.18228 |      0.00969 |
| 3_sobel_loop_unrolling         |     10 |      0.89895 |      0.00957 |
| 4_sobel_loop_fusion            |     10 |      0.80815 |      0.00532 |

### 5) Function inlining
- Remove `pow(..., 2)` function calls and replace with the multiplication of the results
- Replace the `convolution2D` function with the `CONVOLUTION2D` macro

| Executable                     |  #Runs |  Average (s) |  Std Dev (s) |
| ------------------------------ | ------ | ------------ | ------------ |
| 1_sobel_orig                   |     10 |      1.71490 |      0.03946 |
| 2_sobel_loop_interchange       |     10 |      1.18228 |      0.00969 |
| 3_sobel_loop_unrolling         |     10 |      0.89895 |      0.00957 |
| 4_sobel_loop_fusion            |     10 |      0.80815 |      0.00532 |
| 5_sobel_function_inlining      |     10 |      0.20782 |      0.00059 |
