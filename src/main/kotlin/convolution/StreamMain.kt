@file:OptIn(kotlinx.cli.ExperimentalCli::class)

package convolution

import java.nio.file.Path
import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.cli.optional
import kotlin.system.exitProcess

private data class StreamCliConfig(
    val inputDirectory: Path,
    val outputDirectory: Path,
    val kernelNames: String,
    val composeKernels: Boolean,
    val options: StreamingConvolutionOptions,
)

fun main(args: Array<String>) {
    val config = parseStreamArgs(args)

    val exitCode = runCatching {
        val requestedKernels = Kernels.resolveMany(config.kernelNames)
        val effectiveKernels = KernelComposition.prepare(requestedKernels, config.composeKernels)
        val stats = StreamingConvolution.processDirectory(
            inputDirectory = config.inputDirectory,
            outputDirectory = config.outputDirectory,
            kernels = effectiveKernels,
            options = config.options,
        )

        println("Requested kernels: ${requestedKernels.joinToString(",") { it.name }}")
        if (config.composeKernels && requestedKernels.size > 1) {
            val composedKernel = effectiveKernels.single()
            println("Applied as composed kernel: ${composedKernel.name} (${composedKernel.size}x${composedKernel.size})")
        } else {
            println("Applied kernels: ${effectiveKernels.joinToString(",") { it.name }}")
        }
        println("Input directory: ${config.inputDirectory}")
        println("Output directory: ${config.outputDirectory}")
        println("Convolution mode: ${config.options.convolutionMode.cliName}")
        println(
            "Pipeline workers: ${config.options.convolutionWorkers}, " +
                "read queue: ${config.options.readQueueCapacity}, " +
                "write queue: ${config.options.writeQueueCapacity}"
        )
        if (config.options.convolutionMode == ExecutionMode.PARALLEL) {
            println(
                "Inner parallel strategy: ${config.options.parallelOptions.strategy.cliName}, " +
                    "threads: ${config.options.parallelOptions.threads}, " +
                    streamGridDescription(config.options.parallelOptions)
            )
        }
        println(
            "Processed images: discovered=${stats.discoveredImages}, " +
                "read=${stats.readImages}, processed=${stats.processedImages}, written=${stats.writtenImages}"
        )
    }.fold(
        onSuccess = { 0 },
        onFailure = { error ->
            System.err.println(error.message ?: error.toString())
            1
        },
    )

    exitProcess(exitCode)
}

private fun parseStreamArgs(args: Array<String>): StreamCliConfig {
    val defaultWorkers = Runtime.getRuntime().availableProcessors()
    val parser = ArgParser(
        programName = "streamConvolution",
        prefixStyle = ArgParser.OptionPrefixStyle.GNU,
    )
    val inputDirectoryText by parser.argument(
        ArgType.String,
        fullName = "input-dir",
        description = "Input directory with grayscale BMP images",
    )
    val outputDirectoryText by parser.argument(
        ArgType.String,
        fullName = "output-dir",
        description = "Output directory for BMP images",
    )
    val kernelNamesText by parser.argument(
        ArgType.String,
        fullName = "kernels",
        description = "Comma-separated kernel list",
    ).optional()
    val composeKernels by parser.option(
        type = ArgType.Boolean,
        fullName = "compose-kernels",
        description = "Compose several kernels into one larger matrix before execution",
    ).default(false)
    val convolutionModeName by parser.option(
        type = ArgType.Choice(ExecutionMode.entries.map { it.cliName }, { it }),
        fullName = "convolution-mode",
        description = "Sequential or parallel convolution inside each worker",
    ).default(ExecutionMode.SEQUENTIAL.cliName)
    val convolutionWorkers by parser.option(
        type = ArgType.Int,
        fullName = "convolution-workers",
        description = "Number of pipeline workers that process images",
    ).default(defaultWorkers)
    val readQueueCapacity by parser.option(
        type = ArgType.Int,
        fullName = "read-queue-capacity",
        description = "Capacity of the queue between reader and convolution workers",
    ).default(defaultWorkers)
    val writeQueueCapacity by parser.option(
        type = ArgType.Int,
        fullName = "write-queue-capacity",
        description = "Capacity of the queue between convolution workers and writer",
    ).default(defaultWorkers)
    val strategyName by parser.option(
        type = ArgType.Choice(PartitionStrategy.entries.map { it.cliName }, { it }),
        fullName = "strategy",
        description = "Parallel partition strategy for inner convolution",
    ).default(PartitionStrategy.ROWS.cliName)
    val threads by parser.option(
        type = ArgType.Int,
        fullName = "threads",
        description = "Thread count for inner parallel convolution",
    ).default(defaultWorkers)
    val gridText by parser.option(
        type = ArgType.String,
        fullName = "grid",
        description = "Grid partition in ROWSxCOLUMNS form for inner parallel convolution",
    ).default("2x2")
    val tileText by parser.option(
        type = ArgType.String,
        fullName = "tile",
        description = "Tile size in WIDTHxHEIGHT form for inner parallel convolution",
    )

    parser.parse(args)

    val grid = parseAxB(gridText, "grid")
    val tile = tileText?.let { parseAxB(it, "tile") }

    return StreamCliConfig(
        inputDirectory = Path.of(inputDirectoryText),
        outputDirectory = Path.of(outputDirectoryText),
        kernelNames = kernelNamesText ?: "gaussian3",
        composeKernels = composeKernels,
        options = StreamingConvolutionOptions(
            convolutionMode = ExecutionMode.resolve(convolutionModeName),
            convolutionWorkers = convolutionWorkers,
            readQueueCapacity = readQueueCapacity,
            writeQueueCapacity = writeQueueCapacity,
            parallelOptions = ParallelConvolutionOptions(
                strategy = PartitionStrategy.resolve(strategyName),
                threads = threads,
                gridRows = grid.first,
                gridColumns = grid.second,
                tileWidth = tile?.first,
                tileHeight = tile?.second,
            ),
        ),
    )
}

private fun streamGridDescription(options: ParallelConvolutionOptions): String {
    val tileWidth = options.tileWidth
    val tileHeight = options.tileHeight
    return if (tileWidth != null && tileHeight != null) {
        "tile: ${tileWidth}x$tileHeight"
    } else {
        "grid: ${options.gridRows}x${options.gridColumns}"
    }
}
