package convolution

import java.nio.file.Path
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
    val config = parseArgs(args) ?: run {
        printUsage()
        exitProcess(1)
    }

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

private fun parseArgs(args: Array<String>): CliConfig? {
    if (args.size < 2) {
        return null
    }

    val positional = mutableListOf<String>()
    val options = linkedMapOf<String, String>()

    for (arg in args.drop(2)) {
        if (arg.startsWith("--")) {
            val separatorIndex = arg.indexOf('=')
            require(separatorIndex > 2 && separatorIndex < arg.lastIndex) {
                "Options must use --name=value syntax, got '$arg'."
            }
            options[arg.substring(2, separatorIndex)] = arg.substring(separatorIndex + 1)
        } else {
            positional += arg
        }
    }

    require(positional.size <= 1) {
        "Expected at most one positional kernel list, got ${positional.size}."
    }

    val kernelNames = options.remove("kernels")
        ?: positional.firstOrNull()
        ?: "gaussian3"
    val composeKernels = parseBoolean(options.remove("compose-kernels") ?: "false", "compose-kernels")
    val mode = ExecutionMode.resolve(options.remove("mode") ?: ExecutionMode.PARALLEL.cliName)
    val strategy = PartitionStrategy.resolve(options.remove("strategy") ?: PartitionStrategy.ROWS.cliName)
    val threads = options.remove("threads")?.let { parsePositiveInt(it, "threads") }
        ?: Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    val grid = options.remove("grid")?.let { parsePair(it, "grid") } ?: Pair(2, 2)
    val tile = options.remove("tile")?.let { parsePair(it, "tile") }

    require(options.isEmpty()) {
        "Unknown options: ${options.keys.joinToString(", ")}."
    }

    return CliConfig(
        inputPath = Path.of(args[0]),
        outputPath = Path.of(args[1]),
        kernelNames = kernelNames,
        composeKernels = composeKernels,
        mode = mode,
        parallelOptions = ParallelConvolutionOptions(
            strategy = strategy,
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

    return Pair(first, second)
}

private fun parsePositiveInt(value: String, optionName: String): Int {
    val parsed = value.toIntOrNull()
    require(parsed != null) {
        "Option --$optionName must contain an integer value, got '$value'."
    }
    require(parsed > 0) {
        "Option --$optionName must be positive, got '$value'."
    }
    return parsed
}

private fun parseBoolean(value: String, optionName: String): Boolean {
    return when (value.lowercase()) {
        "true" -> true
        "false" -> false
        else -> error("Option --$optionName must be true or false, got '$value'.")
    }
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

private fun printUsage() {
    println(
        """
        Usage:
          ./gradlew run --args="image.bmp output/result.bmp [kernels] [options]"

        Examples:
          ./gradlew run --args="image.bmp output/rows.bmp gaussian3 --mode=parallel --strategy=rows --threads=8"
          ./gradlew run --args="image.bmp output/grid.bmp gaussian3,sharpen3 --strategy=grid --tile=64x64"
          ./gradlew run --args="image.bmp output/composed.bmp gaussian3,sharpen3 --compose-kernels=true"
          ./gradlew run --args="image.bmp output/seq.bmp gaussian3 --mode=sequential"

        Available kernels:
          ${Kernels.availableNames()}

        Options:
          --compose-kernels=true|false
          --mode=${ExecutionMode.availableNames()}
          --strategy=${PartitionStrategy.availableNames()}
          --threads=N
          --grid=ROWSxCOLUMNS
          --tile=WIDTHxHEIGHT
        """.trimIndent()
    )
}
