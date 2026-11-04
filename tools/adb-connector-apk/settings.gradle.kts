// Standalone build, not a module of android/Pipette: this app holds WRITE_SECURE_SETTINGS and
// re-enables wireless debugging, which must never reach the shipping app through a shared source
// set or a merged manifest.
pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "PipetteAdbConnector"

include(":app")
