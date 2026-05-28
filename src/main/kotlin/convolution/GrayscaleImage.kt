package convolution

import org.opencv.core.CvType
import org.opencv.core.Mat

data class GrayscaleImage(
    val width: Int,
    val height: Int,
    val pixels: IntArray,
) {
    init {
        ConvolutionWorker.validateInput(width, height, pixels)
    }
}

object GrayscaleImages {
    fun fromMat(source: Mat): GrayscaleImage {
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

        return GrayscaleImage(width, height, input)
    }

    fun toMat(image: GrayscaleImage): Mat {
        val outputBytes = ByteArray(image.pixels.size) { index ->
            image.pixels[index].toByte()
        }

        return Mat(image.height, image.width, CvType.CV_8UC1).apply {
            put(0, 0, outputBytes)
        }
    }
}
