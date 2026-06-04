# Image Convolution

## Branches

- `main` - initial commit
- `task1` - sequential grayscale bitmap convolution
- `task2` - parallel convolution, kernel composition, benchmarks, CSV tables, SVG plots

## Dependencies

- JDK `23`
- Gradle wrapper `8.10`
- Kotlin `2.1.10`
- OpenCV for JVM: `org.openpnp:opencv:4.9.0-0`
- For plots in `task2`: Python 3, `numpy`, `matplotlib`

## Kernels

- `identity3`
- `box3`
- `gaussian3`
- `sharpen3`

Input images are read as grayscale through OpenCV. `.bmp` is the intended format for the task.

## Task 1

Switch to the branch:

```bash
git checkout task1
```

Run tests:

```bash
./gradlew test
```

Run sequential convolution:

```bash
./gradlew run --args="image.bmp output/result.bmp gaussian3"
./gradlew run --args="image.bmp output/sharpen.bmp sharpen3"
```

## Task 2

Switch to the branch:

```bash
git checkout task2
```

Run tests:

```bash
./gradlew test
```

Run sequential mode:

```bash
./gradlew run --args="image.bmp output/sequential.bmp gaussian3 --mode=sequential"
```

Run parallel mode:

```bash
./gradlew run --args="image.bmp output/rows.bmp gaussian3 --strategy=rows --threads=8"
./gradlew run --args="image.bmp output/columns.bmp gaussian3 --strategy=columns --threads=8"
./gradlew run --args="image.bmp output/pixels.bmp gaussian3 --strategy=pixels --threads=8"
./gradlew run --args="image.bmp output/grid.bmp gaussian3 --strategy=grid --grid=4x4 --threads=8"
./gradlew run --args="image.bmp output/tile.bmp gaussian3 --strategy=grid --tile=64x64 --threads=8"
```

Run a filter pipeline:

```bash
./gradlew run --args="image.bmp output/pipeline.bmp gaussian3,sharpen3"
```

Run mathematical kernel composition:

```bash
./gradlew run --args="image.bmp output/composed-kernel.bmp gaussian3,sharpen3 --compose-kernels=true"
```

## Benchmark Tables And Graphs

Install Python dependencies once:

```bash
python3 -m pip install numpy matplotlib
```

Run the full benchmark:

```bash
./scripts/run_experiment.sh image.bmp
```

Or run benchmark and plotting separately:

```bash
./gradlew benchmarkConvolution --args="image.bmp benchmark-results/results.csv"
python3 scripts/plot_benchmarks.py benchmark-results/results.csv plots/
```

Benchmark results are written as a CSV table. The main time and throughput columns are:

- `avg_ms`
- `min_ms`
- `max_ms`
- `stddev_ms`
- `ci95_low_ms`
- `ci95_high_ms`
- `throughput_mp_s`
- `throughput_ci95_low_mp_s`
- `throughput_ci95_high_mp_s`

The plotting script generates SVG graphs for each kernel pipeline:

- `<kernel-set>-avg-ms.svg`
- `<kernel-set>-throughput.svg`
