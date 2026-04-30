package convolution

import nu.pattern.OpenCV
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.nio.file.Files
import java.nio.file.Path

object OpenCvSupport {
    private var loaded = false

    fun readGrayscale(path: Path): Mat {
        path.parent?.let(Files::createDirectories)
        require(Files.exists(path)) { "Input file does not exist: $path" }
        ensureLoaded()

        val image = Imgcodecs.imread(path.toString(), Imgcodecs.IMREAD_GRAYSCALE)
        require(!image.empty()) { "Failed to read image from $path" }
        return image
    }

    fun write(path: Path, image: Mat) {
        ensureLoaded()
        path.parent?.let(Files::createDirectories)
        check(Imgcodecs.imwrite(path.toString(), image)) { "Failed to write image to $path" }
    }

    private fun ensureLoaded() {
        if (loaded) {
            return
        }

        OpenCV.loadLocally()
        Core.setNumThreads(1)
        loaded = true
    }
}
