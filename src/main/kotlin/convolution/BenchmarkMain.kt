@file:OptIn(kotlinx.cli.ExperimentalCli::class)

package convolution

import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlin.math.sqrt
import kotlin.system.exitProcess
import kotlin.system.measureNanoTime
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVPrinter

private const val WARMUP_ITERATIONS = 5
private const val MEASURED_ITERATIONS = 15
private const val NORMAL_Z_95 = 1.96

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

private data class BenchmarkRow(
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
    val kernelCount: Int,
)

private data class SampleSummary(
    val mean: Double,
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

    val exitCode = runCatching {
        val source = GrayscaleImages.fromMat(OpenCvSupport.readGrayscale(inputPath))
        val rows = runBenchmarks(source)
        writeBenchmarkCsv(outputCsvPath, rows)
        println("Saved benchmark results to $outputCsvPath")
    }.fold(
        onSuccess = { 0 },
        onFailure = { error ->
            System.err.println(error.message ?: error.toString())
            1
        },
    )

    exitProcess(exitCode)
}

private fun parseBenchmarkArgs(args: Array<String>): Pair<Path, Path> {
    val parser = ArgParser(
        programName = "benchmarkConvolution",
        prefixStyle = ArgParser.OptionPrefixStyle.GNU,
    )
    val inputPathText by parser.argument(ArgType.String, fullName = "input", description = "Input grayscale image path")
    val outputCsvPathText by parser.argument(
        ArgType.String,
        fullName = "output-csv",
        description = "Output benchmark CSV path",
    )
    parser.parse(args)
    return Path.of(inputPathText) to Path.of(outputCsvPathText)
}

private fun runBenchmarks(source: GrayscaleImage): List<BenchmarkRow> {
    val rows = mutableListOf<BenchmarkRow>()
    val parallelCases = buildParallelCases()

    for (kernelSet in BENCHMARK_KERNEL_SETS) {
        val kernels = Kernels.resolveMany(kernelSet)
        rows += runBenchmarkCase(source, kernelSet, kernels, parallelOptions = null)
        for (options in parallelCases) {
            rows += runBenchmarkCase(source, kernelSet, kernels, parallelOptions = options)
        }
    }

    return rows
}

private fun buildParallelCases(): List<ParallelConvolutionOptions> {
    val threads = defaultThreadCounts()
    return buildList {
        for (threadCount in threads) {
            add(
                ParallelConvolutionOptions(
                    strategy = PartitionStrategy.PIXELS,
                    threads = threadCount,
                )
            )
            add(
                ParallelConvolutionOptions(
                    strategy = PartitionStrategy.ROWS,
                    threads = threadCount,
                )
            )
            add(
                ParallelConvolutionOptions(
                    strategy = PartitionStrategy.COLUMNS,
                    threads = threadCount,
                )
            )
            for ((gridRows, gridColumns) in BENCHMARK_GRID_SIZES) {
                add(
                    ParallelConvolutionOptions(
                        strategy = PartitionStrategy.GRID,
                        threads = threadCount,
                        gridRows = gridRows,
                        gridColumns = gridColumns,
                    )
                )
            }
            for ((tileWidth, tileHeight) in BENCHMARK_TILE_SIZES) {
                add(
                    ParallelConvolutionOptions(
                        strategy = PartitionStrategy.GRID,
                        threads = threadCount,
                        tileWidth = tileWidth,
                        tileHeight = tileHeight,
                    )
                )
            }
        }
    }
}

private fun defaultThreadCounts(): List<Int> {
    val processors = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    return buildList {
        add(1)
        if (processors >= 2) add(2)
        if (processors >= 4) add(4)
        if (processors >= 8) add(8)
        if (processors !in this) add(processors)
    }.distinct()
}

private fun runBenchmarkCase(
    source: GrayscaleImage,
    kernelSet: String,
    kernels: List<ConvolutionKernel>,
    parallelOptions: ParallelConvolutionOptions?,
): BenchmarkRow {
    repeat(WARMUP_ITERATIONS) {
        runOnce(source, kernels, parallelOptions)
    }

    val timingsMs = DoubleArray(MEASURED_ITERATIONS)
    repeat(MEASURED_ITERATIONS) { iteration ->
        timingsMs[iteration] = measureNanoTime {
            runOnce(source, kernels, parallelOptions)
        } / 1_000_000.0
    }

    val row = summarize(kernelSet, kernels, source, parallelOptions, timingsMs)
    println(
        "Benchmark: kernels=$kernelSet, spec=${displayName(parallelOptions)}, " +
            "threads=${row.threads}, avgMs=${formatDouble(row.avgMs)}"
    )
    return row
}

private fun runOnce(
    source: GrayscaleImage,
    kernels: List<ConvolutionKernel>,
    parallelOptions: ParallelConvolutionOptions?,
) {
    val result = if (parallelOptions == null) {
        var current = source
        for (kernel in kernels) {
            current = SequentialConvolution.apply(current, kernel)
        }
        current
    } else {
        ParallelConvolution.applyImageToImage(
            source = source,
            kernels = kernels,
            options = parallelOptions,
        )
    }

    BenchmarkBlackhole.value = result.pixels[0]
}

private fun summarize(
    kernelSet: String,
    kernels: List<ConvolutionKernel>,
    source: GrayscaleImage,
    parallelOptions: ParallelConvolutionOptions?,
    timingsMs: DoubleArray,
): BenchmarkRow {
    val processedMegapixels = source.width.toDouble() * source.height.toDouble() * kernels.size / 1_000_000.0
    val timeStats = summarizeSamples(timingsMs)
    val throughputSamples = DoubleArray(timingsMs.size) { index ->
        processedMegapixels / (timingsMs[index] / 1_000.0)
    }
    val throughputStats = summarizeSamples(throughputSamples)

    return BenchmarkRow(
        kernelSet = kernelSet,
        mode = modeName(parallelOptions),
        strategy = strategyName(parallelOptions),
        partitionKind = partitionKind(parallelOptions),
        partitionValue = partitionValue(parallelOptions),
        threads = parallelOptions?.threads ?: 1,
        avgMs = timeStats.mean,
        minMs = timeStats.min,
        maxMs = timeStats.max,
        stddevMs = timeStats.stddev,
        ci95LowMs = timeStats.ci95Low,
        ci95HighMs = timeStats.ci95High,
        throughputMpPerSec = throughputStats.mean,
        throughputCi95LowMpPerSec = throughputStats.ci95Low,
        throughputCi95HighMpPerSec = throughputStats.ci95High,
        imageWidth = source.width,
        imageHeight = source.height,
        kernelCount = kernels.size,
    )
}

private fun modeName(parallelOptions: ParallelConvolutionOptions?): String {
    return if (parallelOptions == null) {
        ExecutionMode.SEQUENTIAL.cliName
    } else {
        ExecutionMode.PARALLEL.cliName
    }
}

private fun strategyName(parallelOptions: ParallelConvolutionOptions?): String {
    return parallelOptions?.strategy?.cliName ?: "sequential"
}

private fun partitionKind(parallelOptions: ParallelConvolutionOptions?): String {
    return when {
        parallelOptions == null -> "none"
        parallelOptions.tileWidth != null && parallelOptions.tileHeight != null -> "tile"
        parallelOptions.strategy == PartitionStrategy.GRID -> "grid"
        else -> "none"
    }
}

private fun partitionValue(parallelOptions: ParallelConvolutionOptions?): String {
    return when (partitionKind(parallelOptions)) {
        "tile" -> "${parallelOptions?.tileWidth}x${parallelOptions?.tileHeight}"
        "grid" -> "${parallelOptions?.gridRows}x${parallelOptions?.gridColumns}"
        else -> "-"
    }
}

private fun displayName(parallelOptions: ParallelConvolutionOptions?): String {
    return when (partitionKind(parallelOptions)) {
        "none" -> parallelOptions?.strategy?.cliName ?: "sequential"
        "tile" -> "tile ${parallelOptions?.tileWidth}x${parallelOptions?.tileHeight}"
        "grid" -> "grid ${parallelOptions?.gridRows}x${parallelOptions?.gridColumns}"
        else -> error("Unsupported benchmark case.")
    }
}

private fun writeBenchmarkCsv(outputPath: Path, rows: List<BenchmarkRow>) {
    outputPath.parent?.let(Files::createDirectories)
    Files.newBufferedWriter(outputPath).use { writer ->
        CSVPrinter(writer, CSVFormat.DEFAULT).use { csv ->
            csv.printRecord(
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
                "kernel_count",
            )

            for (row in rows) {
                csv.printRecord(
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
                    row.kernelCount,
                )
            }
        }
    }
}

private fun formatDouble(value: Double): String = String.format(Locale.US, "%.6f", value)

private fun summarizeSamples(samples: DoubleArray): SampleSummary {
    require(samples.isNotEmpty()) {
        "At least one measurement is required."
    }

    val mean = samples.average()
    val min = samples.minOrNull() ?: mean
    val max = samples.maxOrNull() ?: mean
    val stddev = if (samples.size > 1) {
        val variance = samples.sumOf { delta -> (delta - mean) * (delta - mean) } / (samples.size - 1)
        sqrt(variance)
    } else {
        0.0
    }
    val margin = if (samples.size > 1) {
        val sem = stddev / sqrt(samples.size.toDouble())
        NORMAL_Z_95 * sem
    } else {
        0.0
    }

    return SampleSummary(
        mean = mean,
        min = min,
        max = max,
        stddev = stddev,
        ci95Low = mean - margin,
        ci95High = mean + margin,
    )
}
