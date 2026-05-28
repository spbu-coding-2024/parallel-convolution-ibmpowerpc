plugins {
    application
    kotlin("jvm") version "2.1.10"
}

group = "convolution"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.commons:commons-csv:1.14.1")
    implementation("org.jetbrains.kotlinx:kotlinx-cli:0.3.6")
    implementation("org.openpnp:opencv:4.9.0-0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

tasks.register<JavaExec>("benchmarkConvolution") {
    group = "application"
    description = "Run the convolution benchmark matrix and save the results to CSV."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "convolution.BenchmarkMainKt"
}

application {
    mainClass = "convolution.MainKt"
}

kotlin {
    jvmToolchain(23)
}
