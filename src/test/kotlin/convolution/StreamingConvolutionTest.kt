package convolution

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class StreamingConvolutionTest {
    private val imageA = GrayscaleImage(
        width = 4,
        height = 4,
        pixels = intArrayOf(
            10, 20, 30, 40,
            50, 60, 70, 80,
            90, 100, 110, 120,
            130, 140, 150, 160,
        ),
    )

    private val imageB = GrayscaleImage(
        width = 4,
        height = 4,
        pixels = intArrayOf(
            5, 10, 15, 20,
            25, 30, 35, 40,
            45, 50, 55, 60,
            65, 70, 75, 80,
        ),
    )

    private val imageC = GrayscaleImage(
        width = 4,
        height = 4,
        pixels = intArrayOf(
            160, 150, 140, 130,
            120, 110, 100, 90,
            80, 70, 60, 50,
            40, 30, 20, 10,
        ),
    )

    @Test
    fun `streaming pipeline processes multiple images sequentially`() {
        val inputs = linkedMapOf(
            Path.of("a.bmp") to imageA,
            Path.of("b.bmp") to imageB,
            Path.of("c.bmp") to imageC,
        )
        val jobs = inputs.keys.map { path ->
            StreamImageJob(
                inputPath = path,
                outputPath = Path.of("out").resolve(path.fileName.toString()),
            )
        }
        val outputs = linkedMapOf<Path, GrayscaleImage>()
        val kernel = Kernels.resolve("box3")

        val stats = StreamingConvolution.processJobs(
            jobs = jobs,
            kernels = listOf(kernel),
            options = StreamingConvolutionOptions(
                convolutionMode = ExecutionMode.SEQUENTIAL,
                convolutionWorkers = 2,
                readQueueCapacity = 1,
                writeQueueCapacity = 1,
            ),
            reader = { path -> inputs.getValue(path) },
            writer = { path, image -> outputs[path] = image },
        )

        assertEquals(3, stats.discoveredImages)
        assertEquals(3, stats.readImages)
        assertEquals(3, stats.processedImages)
        assertEquals(3, stats.writtenImages)

        for ((inputPath, inputImage) in inputs) {
            val expected = SequentialConvolution.apply(inputImage, kernel)
            val actual = outputs.getValue(Path.of("out").resolve(inputPath.fileName.toString()))
            assertContentEquals(expected.pixels, actual.pixels, "Mismatch for ${inputPath.fileName}")
        }
    }

    @Test
    fun `streaming pipeline supports parallel convolution inside workers`() {
        val inputs = linkedMapOf(
            Path.of("a.bmp") to imageA,
            Path.of("b.bmp") to imageB,
        )
        val jobs = inputs.keys.map { path ->
            StreamImageJob(
                inputPath = path,
                outputPath = Path.of("out").resolve(path.fileName.toString()),
            )
        }
        val outputs = linkedMapOf<Path, GrayscaleImage>()
        val kernels = Kernels.resolveMany("gaussian3,sharpen3")
        val parallelOptions = ParallelConvolutionOptions(
            strategy = PartitionStrategy.ROWS,
            threads = 2,
        )

        StreamingConvolution.processJobs(
            jobs = jobs,
            kernels = kernels,
            options = StreamingConvolutionOptions(
                convolutionMode = ExecutionMode.PARALLEL,
                convolutionWorkers = 2,
                readQueueCapacity = 2,
                writeQueueCapacity = 2,
                parallelOptions = parallelOptions,
            ),
            reader = { path -> inputs.getValue(path) },
            writer = { path, image -> outputs[path] = image },
        )

        for ((inputPath, inputImage) in inputs) {
            val expected = ParallelConvolution.applyImageToImage(inputImage, kernels, parallelOptions)
            val actual = outputs.getValue(Path.of("out").resolve(inputPath.fileName.toString()))
            assertContentEquals(expected.pixels, actual.pixels, "Mismatch for ${inputPath.fileName}")
        }
    }
}
