package convolution

import org.opencv.core.Mat

object SequentialConvolution {
    fun apply(source: GrayscaleImage, kernel: ConvolutionKernel): GrayscaleImage {
        val output = applyAux(source.width, source.height, source.pixels, kernel)
        return source.copy(pixels = output)
    }

    fun applyAux(width: Int, height: Int, input: IntArray, kernel: ConvolutionKernel): IntArray {
        ConvolutionWorker.validateInput(width, height, input)
        val output = IntArray(input.size)

        ConvolutionWorker.convolveRegion(
            width = width,
            height = height,
            input = input,
            output = output,
            kernel = kernel,
            xStart = 0,
            xEnd = width,
            yStart = 0,
            yEnd = height,
        )
        return output
    }
}
