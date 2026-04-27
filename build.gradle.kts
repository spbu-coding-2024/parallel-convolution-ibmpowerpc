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
    implementation("org.openpnp:opencv:4.9.0-0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

application {
    mainClass = "convolution.MainKt"
}

kotlin {
    jvmToolchain(23)
}
