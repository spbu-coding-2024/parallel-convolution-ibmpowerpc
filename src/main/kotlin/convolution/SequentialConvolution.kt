package convolution

import kotlin.math.roundToInt
import org.opencv.core.CvType
import org.opencv.core.Mat

object SequentialConvolution {
    fun apply(source: Mat, kernel: ConvolutionKernel): Mat {
        require(source.type() == CvType.CV_8UC1) {
            "Expected a grayscale CV_8UC1 image, got type ${source.type()}."
        }

        val width = source.cols()
        val height = source.rows()
        val inputBytes = ByteArray(width * height)
        source.get(0, 0, inputBytes)

        val input = IntArray(inputBytes.size) { index ->
            inputBytes[index].toInt() and 0xFF
        }

        val output = apply_aux(width, height, input, kernel)
        val outputBytes = ByteArray(output.size) { index -> output[index].toByte() }

        return Mat(height, width, CvType.CV_8UC1).apply {
            put(0, 0, outputBytes)
        }
    }

    fun apply_aux(width: Int, height: Int, input: IntArray, kernel: ConvolutionKernel): IntArray {
        require(width > 0) { "Image width must be positive." }
        require(height > 0) { "Image height must be positive." }
        require(input.size == width * height) {
            "Input array size ${input.size} does not match image shape ${width}x$height."
        }

        val radius = kernel.size / 2
        val output = IntArray(input.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0.0
                var kernelIndex = 0

                for (kernelY in -radius..radius) {
                    val sourceY = (y + kernelY).coerceIn(0, height - 1)
                    val rowOffset = sourceY * width

                    for (kernelX in -radius..radius) {
                        val sourceX = (x + kernelX).coerceIn(0, width - 1)
                        val pixel = input[rowOffset + sourceX]
                        sum += pixel * kernel.coefficients[kernelIndex++]
                    }
                }

                output[y * width + x] = sum.roundToInt().coerceIn(0, 255)
            }
        }

        return output
    }
}
