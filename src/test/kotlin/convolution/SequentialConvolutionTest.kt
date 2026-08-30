package convolution

import kotlin.test.Test
import kotlin.test.assertContentEquals

class SequentialConvolutionTest {
    @Test
    fun `identity kernel keeps the image unchanged`() {
        val input = intArrayOf(
            10, 20, 30,
            40, 50, 60,
            70, 80, 90,
        )

        val actual = SequentialConvolution.apply_aux(
            width = 3,
            height = 3,
            input = input,
            kernel = Kernels.resolve("identity3"),
        )

        assertContentEquals(input, actual)
    }

    @Test
    fun `box blur uses replicated borders`() {
        val input = intArrayOf(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        )

        val actual = SequentialConvolution.apply_aux(
            width = 3,
            height = 3,
            input = input,
            kernel = Kernels.resolve("box3"),
        )

        val expected = intArrayOf(
            2, 3, 4,
            4, 5, 6,
            6, 7, 8,
        )

        assertContentEquals(expected, actual)
    }
}
