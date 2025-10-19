### Usage
To run the sobel operator algorithm, use the `run.sh` helper script. Use `run.sh --help` to see all options the script offers. Example usage:

```cmd
./run.sh --execution-method=normal --calculate-average=true --times=5
```

The output of the runs are saved inside the `./metrics/` directory categorized by the image size, the optimization level and the execution method. For example, a run using an image of `4096x4096`, `-O3` optimization level and normal execution method, will be in the directory `metrics/O3/normal/`.

### Plot averages
The `average.py` script will look into the `metrics/` directory for "normal" execution methods to plot the average runtimes and standard deviations. You can run the script with the command:

```
python3 average.py
```

> **Note:** the command `run.sh --calculate-average=true` will also use the `average.py` script to calculate the averages.

### Add new image
To add a new image to the test bench, you can use the `generate-golden.sh` script. Use this script by first naming a png file in the format of `SIZE-NAME.png` and then pass it as an argument:

```cmd
./generate-golden.sh 99999-shibuya.png
```

> **Note:** the `SIZE` is both the width and the height of the image. The algorithm only accepts square images.

This script will generate a `.grey` image, put it in the `src/input/` directory, run the golden version algorithm (original) once and use the output as a golden version of the image in the `src/golden/` directory.
