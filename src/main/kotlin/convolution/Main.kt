package convolution

import java.nio.file.Path
import kotlin.system.exitProcess

private data class CliConfig(
    val inputPath: Path,
    val outputPath: Path,
    val kernelName: String,
)

fun main(args: Array<String>) {
    val config = parseArgs(args) ?: run {
        printUsage()
        exitProcess(1)
    }

    val exitCode = runCatching {
        val kernel = Kernels.resolve(config.kernelName)
        val source = OpenCvSupport.readGrayscale(config.inputPath)
        val result = SequentialConvolution.apply(source, kernel)
        OpenCvSupport.write(config.outputPath, result)
        println("Saved '${kernel.name}' result to ${config.outputPath}")
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
    if (args.size !in 2..3) {
        return null
    }

    return CliConfig(
        inputPath = Path.of(args[0]),
        outputPath = Path.of(args[1]),
        kernelName = args.getOrElse(2) { "gaussian3" },
    )
}

private fun printUsage() {
    println(
        """
        Usage:
          ./gradlew run --args="input/source.bmp output/result.bmp [kernel]"

        Available kernels:
          ${Kernels.availableNames()}

        Example:
          ./gradlew run --args="input/source.bmp output/result.bmp sharpen3"
        """.trimIndent()
    )
}
