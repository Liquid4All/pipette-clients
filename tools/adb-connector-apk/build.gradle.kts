// Pinned literally, not via android/Pipette's catalog, so a lab utility is not coupled to the
// app's upgrades. Keep it equal to that catalog's `agp` so both builds want the same Gradle and JDK.
plugins {
    id("com.android.application") version "9.2.1" apply false
}
