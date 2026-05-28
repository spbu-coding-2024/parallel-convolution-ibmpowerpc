package convolution

import kotlin.test.Test
import kotlin.test.assertContentEquals

class ParallelConvolutionTest {
    private val input = intArrayOf(
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200,
    )

    @Test
    fun `all partition strategies match sequential convolution`() {
        val expected = SequentialConvolution.applyAux(
            width = 5,
            height = 4,
            input = input,
            kernel = Kernels.resolve("sharpen3"),
        )

        for (strategy in PartitionStrategy.entries) {
            val actual = ParallelConvolution.apply(
                width = 5,
                height = 4,
                input = input,
                kernel = Kernels.resolve("sharpen3"),
                options = ParallelConvolutionOptions(
                    strategy = strategy,
                    threads = 3,
                    gridRows = 2,
                    gridColumns = 3,
                ),
            )

            assertContentEquals(expected, actual, "Strategy ${strategy.cliName} differs from sequential result.")
        }
    }

    @Test
    fun `grid tile size matches sequential convolution`() {
        val expected = SequentialConvolution.applyAux(
            width = 5,
            height = 4,
            input = input,
            kernel = Kernels.resolve("box3"),
        )

        val actual = ParallelConvolution.apply(
            width = 5,
            height = 4,
            input = input,
            kernel = Kernels.resolve("box3"),
            options = ParallelConvolutionOptions(
                strategy = PartitionStrategy.GRID,
                threads = 4,
                tileWidth = 2,
                tileHeight = 3,
            ),
        )

        assertContentEquals(expected, actual)
    }

    @Test
    fun `multiple kernels are applied sequentially inside parallel pipeline`() {
        val kernels = Kernels.resolveMany("gaussian3,sharpen3")
        val expectedAfterGaussian = SequentialConvolution.applyAux(5, 4, input, kernels[0])
        val expected = SequentialConvolution.applyAux(5, 4, expectedAfterGaussian, kernels[1])

        val actual = ParallelConvolution.applyImageToImage(
            source = GrayscaleImage(width = 5, height = 4, pixels = input),
            kernels = kernels,
            options = ParallelConvolutionOptions(
                strategy = PartitionStrategy.ROWS,
                threads = 2,
            ),
        )

        assertContentEquals(expected, actual.pixels)
    }
}
