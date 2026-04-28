package convolution

enum class ExecutionMode(val cliName: String) {
    SEQUENTIAL("sequential"),
    PARALLEL("parallel");

    companion object {
        fun resolve(name: String): ExecutionMode {
            return entries.firstOrNull { it.cliName == name }
                ?: error("Unknown mode '$name'. Available modes: ${availableNames()}.")
        }

        fun availableNames(): String = entries.joinToString(", ") { it.cliName }
    }
}
