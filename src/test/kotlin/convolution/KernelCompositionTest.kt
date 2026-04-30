package convolution

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class KernelCompositionTest {
    @Test
    fun `box blur composed with box blur produces expected 5x5 kernel`() {
        val composed = KernelComposition.composeAll(Kernels.resolveMany("box3,box3"))

        assertEquals(5, composed.size)
        assertDoubleArrayClose(
            expected = doubleArrayOf(
                1.0 / 81.0, 2.0 / 81.0, 3.0 / 81.0, 2.0 / 81.0, 1.0 / 81.0,
                2.0 / 81.0, 4.0 / 81.0, 6.0 / 81.0, 4.0 / 81.0, 2.0 / 81.0,
                3.0 / 81.0, 6.0 / 81.0, 9.0 / 81.0, 6.0 / 81.0, 3.0 / 81.0,
                2.0 / 81.0, 4.0 / 81.0, 6.0 / 81.0, 4.0 / 81.0, 2.0 / 81.0,
                1.0 / 81.0, 2.0 / 81.0, 3.0 / 81.0, 2.0 / 81.0, 1.0 / 81.0,
            ),
            actual = composed.coefficients,
        )
    }

    @Test
    fun `composed kernel matches unrounded sequential pipeline away from borders`() {
        val input = DoubleArray(81) { index -> ((index * 17 + 13) % 256).toDouble() }
        val kernels = Kernels.resolveMany("gaussian3,sharpen3")

        val afterFirst = convolveToDouble(9, 9, input, kernels[0])
        val expected = convolveToDouble(9, 9, afterFirst, kernels[1])
        val actual = convolveToDouble(9, 9, input, KernelComposition.composeAll(kernels))

        assertInteriorEquals(width = 9, height = 9, radius = 2, expected = expected, actual = actual)
    }

    @Test
    fun `composed kernel matches sequential application on constant image`() {
        val input = IntArray(49) { 42 }
        val kernels = Kernels.resolveMany("gaussian3,sharpen3")

        val afterFirst = SequentialConvolution.applyAux(7, 7, input, kernels[0])
        val expected = SequentialConvolution.applyAux(7, 7, afterFirst, kernels[1])
        val actual = SequentialConvolution.applyAux(7, 7, input, KernelComposition.composeAll(kernels))

        assertContentEquals(expected, actual)
    }

    private fun assertInteriorEquals(
        width: Int,
        height: Int,
        radius: Int,
        expected: DoubleArray,
        actual: DoubleArray,
    ) {
        for (y in radius until height - radius) {
            for (x in radius until width - radius) {
                val index = y * width + x
                val delta = abs(expected[index] - actual[index])
                assertTrue(delta <= 1e-12, "Mismatch at ($x, $y): expected=${expected[index]}, actual=${actual[index]}")
            }
        }
    }

    private fun convolveToDouble(
        width: Int,
        height: Int,
        input: DoubleArray,
        kernel: ConvolutionKernel,
    ): DoubleArray {
        val output = DoubleArray(input.size)
        val radius = kernel.size / 2

        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0.0
                var kernelIndex = 0

                for (kernelY in -radius..radius) {
                    val sourceY = (y + kernelY).coerceIn(0, height - 1)
                    val rowOffset = sourceY * width
                    for (kernelX in -radius..radius) {
                        val sourceX = (x + kernelX).coerceIn(0, width - 1)
                        sum += input[rowOffset + sourceX] * kernel.coefficients[kernelIndex++]
                    }
                }

                output[y * width + x] = sum
            }
        }

        return output
    }

    private fun assertDoubleArrayClose(
        expected: DoubleArray,
        actual: DoubleArray,
        epsilon: Double = 1e-12,
    ) {
        assertEquals(expected.size, actual.size, "Different kernel coefficient count.")
        for (index in expected.indices) {
            val delta = abs(expected[index] - actual[index])
            assertTrue(delta <= epsilon, "Mismatch at coefficient $index: expected=${expected[index]}, actual=${actual[index]}")
        }
    }
}
