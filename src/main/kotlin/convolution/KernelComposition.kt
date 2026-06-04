package convolution

object KernelComposition {
    fun prepare(kernels: List<ConvolutionKernel>, compose: Boolean): List<ConvolutionKernel> {
        require(kernels.isNotEmpty()) { "At least one kernel must be specified." }
        return if (compose && kernels.size > 1) listOf(composeAll(kernels)) else kernels
    }

    fun composeAll(kernels: List<ConvolutionKernel>): ConvolutionKernel {
        require(kernels.isNotEmpty()) { "At least one kernel must be specified." }
        val composed = kernels.reduce(::compose)
        return composed.copy(name = "compose(${kernels.joinToString(",") { it.name }})")
    }

    fun compose(first: ConvolutionKernel, second: ConvolutionKernel): ConvolutionKernel {
        val size = first.size + second.size - 1
        val radius = size / 2
        val firstRadius = first.size / 2
        val secondRadius = second.size / 2
        val coefficients = DoubleArray(size * size)

        for (firstY in 0 until first.size) {
            val firstOffsetY = firstY - firstRadius
            for (firstX in 0 until first.size) {
                val firstCoefficient = first.coefficients[firstY * first.size + firstX]
                if (firstCoefficient == 0.0) {
                    continue
                }

                val firstOffsetX = firstX - firstRadius
                for (secondY in 0 until second.size) {
                    val combinedY = firstOffsetY + (secondY - secondRadius) + radius
                    for (secondX in 0 until second.size) {
                        val combinedX = firstOffsetX + (secondX - secondRadius) + radius
                        val combinedIndex = combinedY * size + combinedX
                        coefficients[combinedIndex] +=
                            firstCoefficient * second.coefficients[secondY * second.size + secondX]
                    }
                }
            }
        }

        return ConvolutionKernel(
            name = "compose(${first.name},${second.name})",
            size = size,
            coefficients = coefficients,
        )
    }
}
