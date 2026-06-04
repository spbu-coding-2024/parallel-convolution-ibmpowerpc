package convolution

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

data class StreamingConvolutionOptions(
    val convolutionMode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    val convolutionWorkers: Int = Runtime.getRuntime().availableProcessors(),
    val readQueueCapacity: Int = Runtime.getRuntime().availableProcessors(),
    val writeQueueCapacity: Int = Runtime.getRuntime().availableProcessors(),
    val parallelOptions: ParallelConvolutionOptions = ParallelConvolutionOptions(),
) {
    init {
        require(convolutionWorkers > 0) { "Convolution worker count must be positive." }
        require(readQueueCapacity > 0) { "Read queue capacity must be positive." }
        require(writeQueueCapacity > 0) { "Write queue capacity must be positive." }
    }
}

data class StreamingConvolutionStats(
    val discoveredImages: Int,
    val readImages: Int,
    val processedImages: Int,
    val writtenImages: Int,
)

internal data class StreamImageJob(
    val inputPath: Path,
    val outputPath: Path,
)

private data class LoadedImageJob(
    val job: StreamImageJob,
    val image: GrayscaleImage,
)

private data class ProcessedImageJob(
    val job: StreamImageJob,
    val image: GrayscaleImage,
)

object StreamingConvolution {
    fun processDirectory(
        inputDirectory: Path,
        outputDirectory: Path,
        kernels: List<ConvolutionKernel>,
        options: StreamingConvolutionOptions,
    ): StreamingConvolutionStats {
        require(Files.isDirectory(inputDirectory)) { "Input directory does not exist: $inputDirectory" }
        Files.createDirectories(outputDirectory)

        val jobs = Files.list(inputDirectory).use { entries ->
            entries
                .filter { Files.isRegularFile(it) }
                .sorted(compareBy { it.fileName.toString() })
                .map { inputPath ->
                    StreamImageJob(
                        inputPath = inputPath,
                        outputPath = outputDirectory.resolve(inputPath.fileName.toString()),
                    )
                }
                .toList()
        }

        require(jobs.isNotEmpty()) {
            "No input files were found in $inputDirectory."
        }

        return processJobs(
            jobs = jobs,
            kernels = kernels,
            options = options,
            reader = { path -> GrayscaleImages.fromMat(OpenCvSupport.readGrayscale(path)) },
            writer = { path, image -> OpenCvSupport.write(path, GrayscaleImages.toMat(image)) },
        )
    }

    internal fun processJobs(
        jobs: List<StreamImageJob>,
        kernels: List<ConvolutionKernel>,
        options: StreamingConvolutionOptions,
        reader: (Path) -> GrayscaleImage,
        writer: (Path, GrayscaleImage) -> Unit,
    ): StreamingConvolutionStats {
        require(jobs.isNotEmpty()) { "At least one image must be provided." }
        require(kernels.isNotEmpty()) { "At least one kernel must be specified." }

        val readQueue = ArrayBlockingQueue<LoadedImageJob>(options.readQueueCapacity)
        val writeQueue = ArrayBlockingQueue<ProcessedImageJob>(options.writeQueueCapacity)
        val readCompleted = AtomicBoolean(false)
        val activeWorkers = AtomicInteger(options.convolutionWorkers)
        val failure = AtomicReference<Throwable?>(null)
        val readCount = AtomicInteger(0)
        val processedCount = AtomicInteger(0)
        val writtenCount = AtomicInteger(0)

        val readerExecutor = Executors.newSingleThreadExecutor()
        val convolutionExecutor = Executors.newFixedThreadPool(options.convolutionWorkers)
        val writerExecutor = Executors.newSingleThreadExecutor()

        try {
            val readerFuture = readerExecutor.submit<Unit> {
                try {
                    for (job in jobs) {
                        if (failure.get() != null) {
                            return@submit
                        }
                        val image = reader(job.inputPath)
                        readCount.incrementAndGet()
                        offerUntilAccepted(readQueue, LoadedImageJob(job, image), failure)
                    }
                } catch (error: Throwable) {
                    failure.compareAndSet(null, error)
                    throw error
                } finally {
                    readCompleted.set(true)
                }
            }

            val workerFutures = List(options.convolutionWorkers) {
                convolutionExecutor.submit<Unit> {
                    try {
                        while (failure.get() == null) {
                            val loaded = pollUntilAvailable(readQueue, readCompleted, failure) ?: return@submit
                            val result = applyConvolution(loaded.image, kernels, options)
                            processedCount.incrementAndGet()
                            offerUntilAccepted(writeQueue, ProcessedImageJob(loaded.job, result), failure)
                        }
                    } catch (error: Throwable) {
                        failure.compareAndSet(null, error)
                        throw error
                    } finally {
                        activeWorkers.decrementAndGet()
                    }
                }
            }

            val writerFuture = writerExecutor.submit<Unit> {
                try {
                    while (failure.get() == null || activeWorkers.get() > 0 || writeQueue.isNotEmpty()) {
                        val processed = pollUntilAvailable(writeQueue, activeWorkers, failure) ?: return@submit
                        writer(processed.job.outputPath, processed.image)
                        writtenCount.incrementAndGet()
                    }
                } catch (error: Throwable) {
                    failure.compareAndSet(null, error)
                    throw error
                }
            }

            awaitAll(listOf(readerFuture, writerFuture) + workerFutures)
            failure.get()?.let { throw it }
        } finally {
            readerExecutor.shutdownNow()
            convolutionExecutor.shutdownNow()
            writerExecutor.shutdownNow()
        }

        return StreamingConvolutionStats(
            discoveredImages = jobs.size,
            readImages = readCount.get(),
            processedImages = processedCount.get(),
            writtenImages = writtenCount.get(),
        )
    }

    private fun applyConvolution(
        image: GrayscaleImage,
        kernels: List<ConvolutionKernel>,
        options: StreamingConvolutionOptions,
    ): GrayscaleImage {
        return when (options.convolutionMode) {
            ExecutionMode.SEQUENTIAL -> {
                var current = image
                for (kernel in kernels) {
                    current = SequentialConvolution.apply(current, kernel)
                }
                current
            }

            ExecutionMode.PARALLEL -> ParallelConvolution.applyImageToImage(
                source = image,
                kernels = kernels,
                options = options.parallelOptions,
            )
        }
    }

    private fun <T> offerUntilAccepted(
        queue: ArrayBlockingQueue<T>,
        item: T,
        failure: AtomicReference<Throwable?>,
    ) {
        while (failure.get() == null) {
            if (queue.offer(item, 100, TimeUnit.MILLISECONDS)) {
                return
            }
        }
    }

    private fun <T> pollUntilAvailable(
        queue: ArrayBlockingQueue<T>,
        producerDone: AtomicBoolean,
        failure: AtomicReference<Throwable?>,
    ): T? {
        while (failure.get() == null) {
            val value = queue.poll(100, TimeUnit.MILLISECONDS)
            if (value != null) {
                return value
            }
            if (producerDone.get()) {
                return null
            }
        }
        return null
    }

    private fun <T> pollUntilAvailable(
        queue: ArrayBlockingQueue<T>,
        activeWorkers: AtomicInteger,
        failure: AtomicReference<Throwable?>,
    ): T? {
        while (failure.get() == null || activeWorkers.get() > 0 || queue.isNotEmpty()) {
            val value = queue.poll(100, TimeUnit.MILLISECONDS)
            if (value != null) {
                return value
            }
            if (activeWorkers.get() == 0) {
                return null
            }
        }
        return null
    }

    private fun awaitAll(futures: List<Future<*>>) {
        var firstFailure: Throwable? = null
        for (future in futures) {
            try {
                future.get()
            } catch (error: Throwable) {
                if (firstFailure == null) {
                    firstFailure = error.cause ?: error
                }
            }
        }
        firstFailure?.let { throw it }
    }
}
