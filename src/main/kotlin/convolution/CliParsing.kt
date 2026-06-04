package convolution

internal fun parseAxB(value: String, optionName: String): Pair<Int, Int> {
    val parts = value.lowercase().split("x")
    require(parts.size == 2) {
        "Option --$optionName must use AxB syntax, got '$value'."
    }

    val first = parts[0].toIntOrNull()
    val second = parts[1].toIntOrNull()
    require(first != null && second != null) {
        "Option --$optionName must contain integer values, got '$value'."
    }
    require(first > 0 && second > 0) {
        "Option --$optionName must contain positive integers, got '$value'."
    }

    return Pair(first, second)
}
