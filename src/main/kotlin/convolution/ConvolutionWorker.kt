package convolution

import kotlin.math.roundToInt

internal object ConvolutionWorker {
    fun validateInput(width: Int, height: Int, input: IntArray) {
        require(width > 0) { "Image width must be positive." }
        require(height > 0) { "Image height must be positive." }
        require(input.size == width * height) {
            "Input array size ${input.size} does not match image shape ${width}x$height."
        }
    }

    fun convolveRegion(
        width: Int,
        height: Int,
        input: IntArray,
        output: IntArray,
        kernel: ConvolutionKernel,
        xStart: Int,
        xEnd: Int,
        yStart: Int,
        yEnd: Int,
    ) {
        for (y in yStart until yEnd) {
            for (x in xStart until xEnd) {
                output[y * width + x] = convolvePixel(width, height, input, kernel, x, y)
            }
        }
    }

    fun convolvePixel(
        width: Int,
        height: Int,
        input: IntArray,
        kernel: ConvolutionKernel,
        x: Int,
        y: Int,
    ): Int {
        val radius = kernel.size / 2
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

        return sum.roundToInt().coerceIn(0, 255)
    }
}
