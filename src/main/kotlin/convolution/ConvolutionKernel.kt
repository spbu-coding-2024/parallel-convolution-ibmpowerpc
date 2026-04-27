package convolution

data class ConvolutionKernel(
    val name: String,
    val size: Int,
    val coefficients: DoubleArray,
) {
    init {
        require(size % 2 == 1) { "Kernel size must be odd." }
        require(coefficients.size == size * size) {
            "Kernel $name must contain exactly ${size * size} coefficients."
        }
    }
}

object Kernels {
    private val all = listOf(
        ConvolutionKernel(
            name = "identity3",
            size = 3,
            coefficients = doubleArrayOf(
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ),
        ),
        ConvolutionKernel(
            name = "box3",
            size = 3,
            coefficients = doubleArrayOf(
                1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            ),
        ),
        ConvolutionKernel(
            name = "gaussian3",
            size = 3,
            coefficients = doubleArrayOf(
                1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
                2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
                1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
            ),
        ),
        ConvolutionKernel(
            name = "sharpen3",
            size = 3,
            coefficients = doubleArrayOf(
                0.0, -1.0, 0.0,
                -1.0, 5.0, -1.0,
                0.0, -1.0, 0.0,
            ),
        ),
    )

    private val byName = all.associateBy { it.name }

    fun resolve(name: String): ConvolutionKernel {
        return byName[name] ?: error(
            "Unknown kernel '$name'. Available kernels: ${availableNames()}."
        )
    }

    fun availableNames(): String = all.joinToString(", ") { it.name }
}
