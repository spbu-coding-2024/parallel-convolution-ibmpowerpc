package convolution

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import org.opencv.core.Mat

enum class PartitionStrategy(val cliName: String) {
    PIXELS("pixels"),
    ROWS("rows"),
    COLUMNS("columns"),
    GRID("grid");

    companion object {
        fun resolve(name: String): PartitionStrategy {
            return entries.firstOrNull { it.cliName == name }
                ?: error("Unknown strategy '$name'. Available strategies: ${availableNames()}.")
        }

        fun availableNames(): String = entries.joinToString(", ") { it.cliName }
    }
}

data class ParallelConvolutionOptions(
    val strategy: PartitionStrategy = PartitionStrategy.ROWS,
    val threads: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    val gridRows: Int = 2,
    val gridColumns: Int = 2,
    val tileWidth: Int? = null,
    val tileHeight: Int? = null,
) {
    init {
        require(threads > 0) { "Thread count must be positive." }
        require(gridRows > 0) { "Grid row count must be positive." }
        require(gridColumns > 0) { "Grid column count must be positive." }
        require((tileWidth == null) == (tileHeight == null)) {
            "Tile width and tile height must be specified together."
        }
        tileWidth?.let { require(it > 0) { "Tile width must be positive." } }
        tileHeight?.let { require(it > 0) { "Tile height must be positive." } }
    }
}

object ParallelConvolution {
    fun applyMatToMat(
        source: Mat,
        kernels: List<ConvolutionKernel>,
        options: ParallelConvolutionOptions,
    ): Mat {
        return GrayscaleImages.toMat(applyImageToImage(GrayscaleImages.fromMat(source), kernels, options))
    }

    fun applyImageToImage(
        source: GrayscaleImage,
        kernels: List<ConvolutionKernel>,
        options: ParallelConvolutionOptions,
    ): GrayscaleImage {
        require(kernels.isNotEmpty()) { "At least one kernel must be specified." }

        var current = source.pixels
        for (kernel in kernels) {
            current = apply(source.width, source.height, current, kernel, options)
        }

        return source.copy(pixels = current)
    }

    fun apply(
        width: Int,
        height: Int,
        input: IntArray,
        kernel: ConvolutionKernel,
        options: ParallelConvolutionOptions,
    ): IntArray {
        ConvolutionWorker.validateInput(width, height, input)

        val output = IntArray(input.size)
        when (options.strategy) {
            PartitionStrategy.PIXELS -> convolveByPixels(width, height, input, output, kernel, options.threads)
            PartitionStrategy.ROWS -> convolveByRows(width, height, input, output, kernel, options.threads)
            PartitionStrategy.COLUMNS -> convolveByColumns(width, height, input, output, kernel, options.threads)
            PartitionStrategy.GRID -> convolveByGrid(width, height, input, output, kernel, options)
        }

        return output
    }

    private fun convolveByPixels(
        width: Int,
        height: Int,
        input: IntArray,
        output: IntArray,
        kernel: ConvolutionKernel,
        threads: Int,
    ) {
        val nextPixel = AtomicInteger(0)
        runWorkers(threads) {
            var index = nextPixel.getAndIncrement()
            while (index < input.size) {
                val y = index / width
                val x = index - y * width
                output[index] = ConvolutionWorker.convolvePixel(width, height, input, kernel, x, y)
                index = nextPixel.getAndIncrement()
            }
        }
    }

    private fun convolveByRows(
        width: Int,
        height: Int,
        input: IntArray,
        output: IntArray,
        kernel: ConvolutionKernel,
        threads: Int,
    ) {
        val nextRow = AtomicInteger(0)
        runWorkers(threads) {
            var y = nextRow.getAndIncrement()
            while (y < height) {
                ConvolutionWorker.convolveRegion(width, height, input, output, kernel, 0, width, y, y + 1)
                y = nextRow.getAndIncrement()
            }

        }
    }

    private fun convolveByColumns(
        width: Int,
        height: Int,
        input: IntArray,
        output: IntArray,
        kernel: ConvolutionKernel,
        threads: Int,
    ) {
        val nextColumn = AtomicInteger(0)
        runWorkers(threads) {
            var x = nextColumn.getAndIncrement()
            while (x < width) {
                ConvolutionWorker.convolveRegion(width, height, input, output, kernel, x, x + 1, 0, height)
                x = nextColumn.getAndIncrement()
            }
        }
    }

    private fun convolveByGrid(
        width: Int,
        height: Int,
        input: IntArray,
        output: IntArray,
        kernel: ConvolutionKernel,
        options: ParallelConvolutionOptions,
    ) {
        val regions = buildGridRegions(width, height, options)
        val nextRegion = AtomicInteger(0)

        runWorkers(options.threads) {
            var index = nextRegion.getAndIncrement()
            while (index < regions.size) {
                val region = regions[index]
                ConvolutionWorker.convolveRegion(
                    width = width,
                    height = height,
                    input = input,
                    output = output,
                    kernel = kernel,
                    xStart = region.xStart,
                    xEnd = region.xEnd,
                    yStart = region.yStart,
                    yEnd = region.yEnd,
                )
                index = nextRegion.getAndIncrement()
            }
        }
    }

    private fun runWorkers(threads: Int, work: () -> Unit) {
        val executor = Executors.newFixedThreadPool(threads)
        try {
            val futures = List(threads) {
                executor.submit { work() }
            }
            for (future in futures) {
                future.get()
            }
        } finally {
            executor.shutdown()
        }
    }

    private fun buildGridRegions(
        width: Int,
        height: Int,
        options: ParallelConvolutionOptions,
    ): List<Region> {
        val tileWidth = options.tileWidth
        val tileHeight = options.tileHeight

        if (tileWidth != null && tileHeight != null) {
            return buildList {
                var y = 0
                while (y < height) {
                    val yEnd = (y + tileHeight).coerceAtMost(height)
                    var x = 0
                    while (x < width) {
                        val xEnd = (x + tileWidth).coerceAtMost(width)
                        add(Region(x, xEnd, y, yEnd))
                        x += tileWidth
                    }
                    y += tileHeight
                }
            }
        }

        return buildList {
            for (gridY in 0 until options.gridRows) {
                val yStart = gridY * height / options.gridRows
                val yEnd = (gridY + 1) * height / options.gridRows

                for (gridX in 0 until options.gridColumns) {
                    val xStart = gridX * width / options.gridColumns
                    val xEnd = (gridX + 1) * width / options.gridColumns

                    if (xStart < xEnd && yStart < yEnd) {
                        add(Region(xStart, xEnd, yStart, yEnd))
                    }
                }
            }
        }
    }

    private data class Region(
        val xStart: Int,
        val xEnd: Int,
        val yStart: Int,
        val yEnd: Int,
    )
}
