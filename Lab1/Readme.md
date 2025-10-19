# Code Optimizations
The optimizations made to the code are noted here. Each version has all the previous optimizations plus the new ones. All runs were made on the same performance core of a 12th Gen Intel® Core™ i5-1235U × 12 CPU without any compiler optimizations (`-O0` option).

### How to run
To run the program, use the `run.sh` helper script. Use `run.sh --help` to see all options the script offers. Example usage:

```cmd
./run.sh --execution-method=normal --calculate-average=true --times=5
```

The output of the runs are saved inside the `./metrics/` directory categorized by the execution method.

### Plot averages
To plot the averages and standard deviation, run the following commands:

```cmd
python3 -m venv ./venv
```

```cmd
pip install -r requirements.txt
```

Now the plot will be generated every time the `run.sh` script is executed, or you can manually run the `average.py` script:

```
python3 average.py
```

> **Note:** The `average.py` script gets the numbers from 

### 1) Original algorithm
- Golden version 🪙️
- Sloooow 🐌️


### 2) Loop interchange
- Switch the loops from column first to row first for better cache locality on the following loops:
  - The `convolution2D` function loop
  - The `sobel` function main loop that applies the filter
  - The `sobel` function loop that calculates the PSNR


### 3) Loop unrolling
- Fully unroll the `convolution2D` function loop
- Unroll the main main loop and the PSNR calculation loop with a factor of `4`, handle left over columns with a `switch` statement


### 4) Loop fusion
- Fuse together the PSNR calculation and the sobel operator


### 5) Function inlining
- Replace the `convolution2D` function with the `CONVOLUTION2D` macro


### 6) Common subexpression elimination
- Minimize duplicate computations by introducing `i_times_SIZE` and `i_times_SIZE_plus_j` variables


### 7) Strength reduction
- Remove `pow(..., 2)` function calls and replace with the multiplication of the results
- Increment `i_times_SIZE` at the end of the loop instead of multiplying
- TODO: do binary AND between 256 and `res` instead of comparison

### 10) Referencing
- Reference offset arrays to eliminate redundant operations `input[offset + i]` becomes `ofst_input = &input[offset]` and is then indexed: `ofst_input[i]`
- Perform strength reduction on indexing operations and incrementing variables