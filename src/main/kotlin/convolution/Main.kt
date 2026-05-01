@file:OptIn(kotlinx.cli.ExperimentalCli::class)

package convolution

import java.nio.file.Path
import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.cli.optional
import kotlin.system.exitProcess

private data class CliConfig(
    val inputPath: Path,
    val outputPath: Path,
    val kernelNames: String,
    val composeKernels: Boolean,
    val mode: ExecutionMode,
    val parallelOptions: ParallelConvolutionOptions,
)

fun main(args: Array<String>) {
    val config = parseArgs(args)

    val exitCode = runCatching {
        val requestedKernels = Kernels.resolveMany(config.kernelNames)
        val effectiveKernels = KernelComposition.prepare(requestedKernels, config.composeKernels)
        val source = OpenCvSupport.readGrayscale(config.inputPath)
        val result = when (config.mode) {
            ExecutionMode.SEQUENTIAL -> applySequential(source = source, kernels = effectiveKernels)
            ExecutionMode.PARALLEL -> ParallelConvolution.applyMatToMat(
                source = source,
                kernels = effectiveKernels,
                options = config.parallelOptions,
            )
        }

        OpenCvSupport.write(config.outputPath, result)
        println("Requested kernels: ${requestedKernels.joinToString(",") { it.name }}")
        if (config.composeKernels && requestedKernels.size > 1) {
            val composedKernel = effectiveKernels.single()
            println("Applied as composed kernel: ${composedKernel.name} (${composedKernel.size}x${composedKernel.size})")
            println("Note: composed-kernel output can differ from the multi-pass pipeline because each regular pass rounds pixels and reapplies clamp borders.")
        } else {
            println("Applied kernels: ${effectiveKernels.joinToString(",") { it.name }}")
        }
        println("Saved result to ${config.outputPath}")
        println("Mode: ${config.mode.cliName}")
        if (config.mode == ExecutionMode.PARALLEL) {
            println(
                "Parallel strategy: ${config.parallelOptions.strategy.cliName}, " +
                    "threads: ${config.parallelOptions.threads}, " +
                    gridDescription(config.parallelOptions)
            )
        }
    }.fold(
        onSuccess = { 0 },
        onFailure = { error ->
            System.err.println(error.message ?: error.toString())
            1
        },
    )

    exitProcess(exitCode)
}

private fun parseArgs(args: Array<String>): CliConfig {
    val parser = ArgParser(
        programName = "convolution",
        prefixStyle = ArgParser.OptionPrefixStyle.GNU,
    )
    val inputPathText by parser.argument(ArgType.String, fullName = "input", description = "Input grayscale image path")
    val outputPathText by parser.argument(ArgType.String, fullName = "output", description = "Output image path")
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
    val modeName by parser.option(
        type = ArgType.Choice(ExecutionMode.entries.map { it.cliName }, { it }),
        fullName = "mode",
        description = "Execution mode",
    ).default(ExecutionMode.PARALLEL.cliName)
    val strategyName by parser.option(
        type = ArgType.Choice(PartitionStrategy.entries.map { it.cliName }, { it }),
        fullName = "strategy",
        description = "Parallel partition strategy",
    ).default(PartitionStrategy.ROWS.cliName)
    val threads by parser.option(
        type = ArgType.Int,
        fullName = "threads",
        description = "Thread count",
    ).default(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
    val gridText by parser.option(
        type = ArgType.String,
        fullName = "grid",
        description = "Grid partition in ROWSxCOLUMNS form",
    ).default("2x2")
    val tileText by parser.option(
        type = ArgType.String,
        fullName = "tile",
        description = "Tile size in WIDTHxHEIGHT form",
    )

    parser.parse(args)

    val grid = parsePair(gridText, "grid")
    val tile = tileText?.let { parsePair(it, "tile") }

    return CliConfig(
        inputPath = Path.of(inputPathText),
        outputPath = Path.of(outputPathText),
        kernelNames = kernelNamesText ?: "gaussian3",
        composeKernels = composeKernels,
        mode = ExecutionMode.resolve(modeName),
        parallelOptions = ParallelConvolutionOptions(
            strategy = PartitionStrategy.resolve(strategyName),
            threads = threads,
            gridRows = grid.first,
            gridColumns = grid.second,
            tileWidth = tile?.first,
            tileHeight = tile?.second,
        ),
    )
}

private fun applySequential(source: org.opencv.core.Mat, kernels: List<ConvolutionKernel>): org.opencv.core.Mat {
    var current = GrayscaleImages.fromMat(source)
    for (kernel in kernels) {
        current = SequentialConvolution.apply(current, kernel)
    }
    return GrayscaleImages.toMat(current)
}

private fun parsePair(value: String, optionName: String): Pair<Int, Int> {
    val parts = value.lowercase().split("x")
    require(parts.size == 2) {
        "Option --$optionName must use AxB syntax, got '$value'."
    }

    val first = parts[0].toIntOrNull()
    val second = parts[1].toIntOrNull()
    require(first != null && second != null) {
        "Option --$optionName must contain integer values, got '$value'."
    }
    require(first > 0 && second > 0) {
        "Option --$optionName must contain positive integers, got '$value'."
    }

    return Pair(first, second)
}

private fun gridDescription(options: ParallelConvolutionOptions): String {
    val tileWidth = options.tileWidth
    val tileHeight = options.tileHeight
    return if (tileWidth != null && tileHeight != null) {
        "tile: ${tileWidth}x$tileHeight"
    } else {
        "grid: ${options.gridRows}x${options.gridColumns}"
    }
}
