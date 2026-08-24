import com.vanniktech.maven.publish.JavaLibrary
import com.vanniktech.maven.publish.JavadocJar
import com.vanniktech.maven.publish.SourcesJar

plugins {
    `java-library`
    jacoco
    id("com.diffplug.spotless") version "6.25.0"
    id("me.champeau.jmh") version "0.7.2"
    id("com.vanniktech.maven.publish") version "0.37.0"
}

group = property("group") as String
version = property("version") as String

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

val demo: SourceSet by sourceSets.creating

dependencies {
    testImplementation(libs.junit)
    testImplementation(libs.xgboost4j)

    jmh(libs.xgboost4j)

    "demoImplementation"(sourceSets.main.get().output)
    "demoImplementation"(libs.xgboost4j)
}

tasks.register<JavaExec>("demo") {
    group = "application"
    description = "Runs the end-to-end Iris train-then-predict tutorial."
    classpath = demo.runtimeClasspath
    mainClass.set("com.yelp.xgboost.demo.IrisDemo")
}

jmh {
    warmupIterations.set(5)
    iterations.set(10)
    fork.set(2)
}

tasks.test {
    useJUnit()
    maxParallelForks = 1
    testLogging {
        events("passed", "skipped", "failed")
    }
    finalizedBy(tasks.jacocoTestReport)
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}

tasks.withType<Javadoc> {
    (options as StandardJavadocDocletOptions).addStringOption("Xdoclint:none", "-quiet")
}

spotless {
    java {
        googleJavaFormat("1.19.2")
        importOrder()
        removeUnusedImports()
        trimTrailingWhitespace()
        endWithNewline()
    }
}

mavenPublishing {
    configure(JavaLibrary(javadocJar = JavadocJar.Javadoc(), sourcesJar = SourcesJar.Sources()))
    publishToMavenCentral()
    if (providers.gradleProperty("signingInMemoryKey").isPresent) {
        signAllPublications()
    }

    coordinates(group as String, "xgboost-predictor", version as String)

    pom {
        name.set("xgboost-predictor")
        description.set("Pure-JVM XGBoost predictor for online inference.")
        url.set("https://github.com/Yelp/xgboost-predictor")
        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
            }
        }
        developers {
            developer {
                id.set("yelp")
                name.set("Yelp")
                organization.set("Yelp Inc.")
                organizationUrl.set("https://github.com/Yelp")
            }
        }
        scm {
            connection.set("scm:git:https://github.com/Yelp/xgboost-predictor.git")
            developerConnection.set("scm:git:ssh://git@github.com/Yelp/xgboost-predictor.git")
            url.set("https://github.com/Yelp/xgboost-predictor")
        }
    }
}
