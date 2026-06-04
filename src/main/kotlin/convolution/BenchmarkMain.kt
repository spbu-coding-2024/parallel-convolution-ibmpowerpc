package convolution

import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.isRegularFile
import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlin.math.sqrt
import kotlin.system.exitProcess
import kotlin.system.measureNanoTime
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVPrinter

private const val WARMUP_ITERATIONS = 5
private const val MEASURED_ITERATIONS = 15
private const val NORMAL_Z_95 = 1.96
private const val WORKLOAD_SINGLE_IMAGE = "single-image"
private const val WORKLOAD_IMAGE_BATCH = "image-batch"

private val BENCHMARK_KERNEL_SETS = listOf(
    "gaussian3",
    "gaussian3,sharpen3",
)

private val BENCHMARK_GRID_SIZES = listOf(
    2 to 2,
    4 to 4,
)

private val BENCHMARK_TILE_SIZES = listOf(
    32 to 32,
    64 to 64,
    128 to 128,
)

private data class BenchmarkInput(
    val singleImage: GrayscaleImage,
    val batchImages: List<GrayscaleImage>,
)

private data class BenchmarkRow(
    val workload: String,
    val kernelSet: String,
    val mode: String,
    val strategy: String,
    val partitionKind: String,
    val partitionValue: String,
    val threads: Int,
    val avgMs: Double,
    val minMs: Double,
    val maxMs: Double,
    val stddevMs: Double,
    val ci95LowMs: Double,
    val ci95HighMs: Double,
    val throughputMpPerSec: Double,
    val throughputCi95LowMpPerSec: Double,
    val throughputCi95HighMpPerSec: Double,
    val imageWidth: Int,
    val imageHeight: Int,
    val imageCount: Int,
    val totalPixels: Long,
    val kernelCount: Int,
)

private data class SampleSummary(
    val average: Double,
    val min: Double,
    val max: Double,
    val stddev: Double,
    val ci95Low: Double,
    val ci95High: Double,
)

private object BenchmarkBlackhole {
    @Volatile
    var value: Int = 0
}

fun main(args: Array<String>) {
    val (inputPath, outputCsvPath) = parseBenchmarkArgs(args)

    runCatching {
        val input = readBenchmarkInput(inputPath)
        val rows = runBenchmarks(input)
        Files.createDirectories(outputCsvPath.parent ?: Path.of("."))
        writeBenchmarkCsv(outputCsvPath, rows)
        println("Benchmark results saved to $outputCsvPath")
    }.onFailure { error ->
        System.err.println(error.message ?: error.toString())
        exitProcess(1)
    }
}

private fun parseBenchmarkArgs(args: Array<String>): Pair<Path, Path> {
    val parser = ArgParser("benchmarkConvolution")
    val input by parser.argument(
        ArgType.String,
        fullName = "input",
        description = "Input grayscale BMP path or a directory with grayscale BMP images.",
    )
    val outputCsv by parser.option(
        ArgType.String,
        fullName = "output-csv",
        shortName = "o",
        description = "Path for the benchmark CSV file.",
    ).default("benchmark-results/convolution-benchmark.csv")

    parser.parse(args)
    return Path.of(input) to Path.of(outputCsv)
}

private fun readBenchmarkInput(inputPath: Path): BenchmarkInput {
    return if (inputPath.isDirectory()) {
        val imagePaths = Files.list(inputPath).use { entries ->
            entries
                .filter { it.isRegularFile() }
                .sorted()
                .toList()
        }
        require(imagePaths.isNotEmpty()) { "Input directory is empty: $inputPath" }
        val images = imagePaths.map { path -> GrayscaleImages.fromMat(OpenCvSupport.readGrayscale(path)) }
        BenchmarkInput(singleImage = images.first(), batchImages = images)
    } else {
        require(inputPath.exists()) { "Input path does not exist: $inputPath" }
        val image = GrayscaleImages.fromMat(OpenCvSupport.readGrayscale(inputPath))
        BenchmarkInput(singleImage = image, batchImages = listOf(image))
    }
}

private fun runBenchmarks(input: BenchmarkInput): List<BenchmarkRow> = buildList {
    val singleImage = input.singleImage
    val batchImages = input.batchImages

    for (kernelSpec in BENCHMARK_KERNEL_SETS) {
        val kernels = Kernels.resolveMany(kernelSpec)

        add(runSingleImageBenchmarkCase(singleImage, kernelSpec, kernels, parallelOptions = null))
        for (parallelOptions in buildSingleImageParallelCases()) {
            add(runSingleImageBenchmarkCase(singleImage, kernelSpec, kernels, parallelOptions))
        }

        add(runImageBatchBenchmarkCase(batchImages, kernelSpec, kernels, imageWorkers = null))
        for (imageWorkers in defaultThreadCounts()) {
            add(runImageBatchBenchmarkCase(batchImages, kernelSpec, kernels, imageWorkers))
        }
    }
}

private fun buildSingleImageParallelCases(): List<ParallelConvolutionOptions> = buildList {
    for (threads in defaultThreadCounts()) {
        add(ParallelConvolutionOptions(PartitionStrategy.PIXELS, threads))
        add(ParallelConvolutionOptions(PartitionStrategy.ROWS, threads))
        add(ParallelConvolutionOptions(PartitionStrategy.COLUMNS, threads))

        for ((columns, rows) in BENCHMARK_GRID_SIZES) {
            add(
                ParallelConvolutionOptions(
                    strategy = PartitionStrategy.GRID,
                    threads = threads,
                    gridColumns = columns,
                    gridRows = rows,
                ),
            )
        }

        for ((tileWidth, tileHeight) in BENCHMARK_TILE_SIZES) {
            add(
                ParallelConvolutionOptions(
                    strategy = PartitionStrategy.GRID,
                    threads = threads,
                    tileWidth = tileWidth,
                    tileHeight = tileHeight,
                ),
            )
        }
    }
}

private fun defaultThreadCounts(): List<Int> {
    val processors = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    return listOf(1, 2, 4, 8, processors)
        .filter { it <= processors }
        .distinct()
}

private fun runSingleImageBenchmarkCase(
    source: GrayscaleImage,
    kernelSpec: String,
    kernels: List<ConvolutionKernel>,
    parallelOptions: ParallelConvolutionOptions?,
): BenchmarkRow {
    repeat(WARMUP_ITERATIONS) {
        runSingleImageOnce(source, kernels, parallelOptions)
    }

    val timingsMs = DoubleArray(MEASURED_ITERATIONS)
    for (iteration in timingsMs.indices) {
        timingsMs[iteration] = measureNanoTime {
            runSingleImageOnce(source, kernels, parallelOptions)
        } / 1_000_000.0
    }

    return summarizeSingleImage(source, kernelSpec, kernels, parallelOptions, timingsMs)
}

private fun runImageBatchBenchmarkCase(
    sources: List<GrayscaleImage>,
    kernelSpec: String,
    kernels: List<ConvolutionKernel>,
    imageWorkers: Int?,
): BenchmarkRow {
    repeat(WARMUP_ITERATIONS) {
        runImageBatchOnce(sources, kernels, imageWorkers)
    }

    val timingsMs = DoubleArray(MEASURED_ITERATIONS)
    for (iteration in timingsMs.indices) {
        timingsMs[iteration] = measureNanoTime {
            runImageBatchOnce(sources, kernels, imageWorkers)
        } / 1_000_000.0
    }

    return summarizeImageBatch(sources, kernelSpec, kernels, imageWorkers, timingsMs)
}

private fun runSingleImageOnce(
    source: GrayscaleImage,
    kernels: List<ConvolutionKernel>,
    parallelOptions: ParallelConvolutionOptions?,
) {
    val result = if (parallelOptions == null) {
        kernels.fold(source) { current, kernel -> SequentialConvolution.apply(current, kernel) }
    } else {
        ParallelConvolution.applyImageToImage(source, kernels, parallelOptions)
    }
    BenchmarkBlackhole.value = result.pixels[result.pixels.size / 2]
}

private fun runImageBatchOnce(
    sources: List<GrayscaleImage>,
    kernels: List<ConvolutionKernel>,
    imageWorkers: Int?,
) {
    if (imageWorkers == null) {
        var blackhole = 0
        for (source in sources) {
            val result = kernels.fold(source) { current, kernel -> SequentialConvolution.apply(current, kernel) }
            blackhole = blackhole xor result.pixels[result.pixels.size / 2]
        }
        BenchmarkBlackhole.value = blackhole
        return
    }

    val imagesByPath = sources.mapIndexed { index, image ->
        Path.of("image-$index.bmp") to image
    }.toMap()
    val jobs = imagesByPath.keys.map { inputPath ->
        StreamImageJob(
            inputPath = inputPath,
            outputPath = Path.of("output").resolve(inputPath.fileName),
        )
    }
    var blackhole = 0

    StreamingConvolution.processJobs(
        jobs = jobs,
        kernels = kernels,
        options = StreamingConvolutionOptions(
            convolutionMode = ExecutionMode.SEQUENTIAL,
            convolutionWorkers = imageWorkers,
            readQueueCapacity = imageWorkers,
            writeQueueCapacity = imageWorkers,
        ),
        reader = { path -> imagesByPath.getValue(path) },
        writer = { _, image ->
            blackhole = blackhole xor image.pixels[image.pixels.size / 2]
        },
    )
    BenchmarkBlackhole.value = blackhole
}

private fun summarizeSingleImage(
    source: GrayscaleImage,
    kernelSpec: String,
    kernels: List<ConvolutionKernel>,
    parallelOptions: ParallelConvolutionOptions?,
    timingsMs: DoubleArray,
): BenchmarkRow {
    val totalPixels = source.width.toLong() * source.height.toLong()
    return summarize(
        workload = WORKLOAD_SINGLE_IMAGE,
        kernelSpec = kernelSpec,
        kernels = kernels,
        timingsMs = timingsMs,
        mode = modeName(parallelOptions),
        strategy = strategyName(parallelOptions),
        partitionKind = partitionKind(parallelOptions),
        partitionValue = partitionValue(parallelOptions),
        threads = parallelOptions?.threads ?: 1,
        imageWidth = source.width,
        imageHeight = source.height,
        imageCount = 1,
        totalPixels = totalPixels,
    )
}

private fun summarizeImageBatch(
    sources: List<GrayscaleImage>,
    kernelSpec: String,
    kernels: List<ConvolutionKernel>,
    imageWorkers: Int?,
    timingsMs: DoubleArray,
): BenchmarkRow {
    val firstImage = sources.first()
    val totalPixels = sources.sumOf { it.width.toLong() * it.height.toLong() }
    return summarize(
        workload = WORKLOAD_IMAGE_BATCH,
        kernelSpec = kernelSpec,
        kernels = kernels,
        timingsMs = timingsMs,
        mode = if (imageWorkers == null) "sequential" else "parallel",
        strategy = if (imageWorkers == null) "sequential" else "images",
        partitionKind = "none",
        partitionValue = "-",
        threads = imageWorkers ?: 1,
        imageWidth = firstImage.width,
        imageHeight = firstImage.height,
        imageCount = sources.size,
        totalPixels = totalPixels,
    )
}

private fun summarize(
    workload: String,
    kernelSpec: String,
    kernels: List<ConvolutionKernel>,
    timingsMs: DoubleArray,
    mode: String,
    strategy: String,
    partitionKind: String,
    partitionValue: String,
    threads: Int,
    imageWidth: Int,
    imageHeight: Int,
    imageCount: Int,
    totalPixels: Long,
): BenchmarkRow {
    val timeSummary = summarizeSamples(timingsMs)
    val processedMegapixels = totalPixels * kernels.size / 1_000_000.0
    val throughputSamples = timingsMs.map { sampleMs ->
        processedMegapixels / (sampleMs / 1_000.0)
    }.toDoubleArray()
    val throughputSummary = summarizeSamples(throughputSamples)

    return BenchmarkRow(
        workload = workload,
        kernelSet = displayName(kernelSpec, kernels),
        mode = mode,
        strategy = strategy,
        partitionKind = partitionKind,
        partitionValue = partitionValue,
        threads = threads,
        avgMs = timeSummary.average,
        minMs = timeSummary.min,
        maxMs = timeSummary.max,
        stddevMs = timeSummary.stddev,
        ci95LowMs = timeSummary.ci95Low,
        ci95HighMs = timeSummary.ci95High,
        throughputMpPerSec = throughputSummary.average,
        throughputCi95LowMpPerSec = throughputSummary.ci95Low,
        throughputCi95HighMpPerSec = throughputSummary.ci95High,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        imageCount = imageCount,
        totalPixels = totalPixels,
        kernelCount = kernels.size,
    )
}

private fun modeName(parallelOptions: ParallelConvolutionOptions?): String =
    if (parallelOptions == null) "sequential" else "parallel"

private fun strategyName(parallelOptions: ParallelConvolutionOptions?): String =
    parallelOptions?.strategy?.cliName ?: "sequential"

private fun partitionKind(parallelOptions: ParallelConvolutionOptions?): String = when {
    parallelOptions == null -> "none"
    parallelOptions.tileWidth != null && parallelOptions.tileHeight != null -> "tile"
    parallelOptions.strategy == PartitionStrategy.GRID -> "grid"
    else -> "none"
}

private fun partitionValue(parallelOptions: ParallelConvolutionOptions?): String = when (partitionKind(parallelOptions)) {
    "tile" -> "${parallelOptions?.tileWidth}x${parallelOptions?.tileHeight}"
    "grid" -> "${parallelOptions?.gridRows}x${parallelOptions?.gridColumns}"
    else -> "-"
}

private fun displayName(kernelSpec: String, kernels: List<ConvolutionKernel>): String =
    if (kernels.size == 1) kernels.single().name else kernelSpec

private fun writeBenchmarkCsv(path: Path, rows: List<BenchmarkRow>) {
    Files.newBufferedWriter(path).use { writer ->
        CSVPrinter(
            writer,
            CSVFormat.DEFAULT.builder()
                .setHeader(
                    "workload",
                    "kernel_set",
                    "mode",
                    "strategy",
                    "partition_kind",
                    "partition_value",
                    "threads",
                    "avg_ms",
                    "min_ms",
                    "max_ms",
                    "stddev_ms",
                    "ci95_low_ms",
                    "ci95_high_ms",
                    "throughput_mp_s",
                    "throughput_ci95_low_mp_s",
                    "throughput_ci95_high_mp_s",
                    "image_width",
                    "image_height",
                    "image_count",
                    "total_pixels",
                    "kernel_count",
                )
                .get(),
        ).use { printer ->
            for (row in rows) {
                printer.printRecord(
                    row.workload,
                    row.kernelSet,
                    row.mode,
                    row.strategy,
                    row.partitionKind,
                    row.partitionValue,
                    row.threads,
                    formatDouble(row.avgMs),
                    formatDouble(row.minMs),
                    formatDouble(row.maxMs),
                    formatDouble(row.stddevMs),
                    formatDouble(row.ci95LowMs),
                    formatDouble(row.ci95HighMs),
                    formatDouble(row.throughputMpPerSec),
                    formatDouble(row.throughputCi95LowMpPerSec),
                    formatDouble(row.throughputCi95HighMpPerSec),
                    row.imageWidth,
                    row.imageHeight,
                    row.imageCount,
                    row.totalPixels,
                    row.kernelCount,
                )
            }
        }
    }
}

private fun formatDouble(value: Double): String = String.format(Locale.US, "%.6f", value)

private fun summarizeSamples(samples: DoubleArray): SampleSummary {
    val average = samples.average()
    val min = samples.minOrNull() ?: 0.0
    val max = samples.maxOrNull() ?: 0.0
    val variance = if (samples.size > 1) {
        samples.sumOf { value ->
            val delta = value - average
            delta * delta
        } / (samples.size - 1)
    } else {
        0.0
    }
    val stddev = sqrt(variance)
    val margin = if (samples.size > 1) {
        val sem = stddev / sqrt(samples.size.toDouble())
        NORMAL_Z_95 * sem
    } else {
        0.0
    }

    return SampleSummary(
        average = average,
        min = min,
        max = max,
        stddev = stddev,
        ci95Low = average - margin,
        ci95High = average + margin,
    )
}
