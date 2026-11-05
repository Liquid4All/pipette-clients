package ai.liquid.pipette.thermal

import kotlin.math.roundToInt

enum class AndroidThermalStatus(val wireValue: String) {
  NONE("none"),
  LIGHT("light"),
  MODERATE("moderate"),
  SEVERE("severe"),
  CRITICAL("critical"),
  EMERGENCY("emergency"),
  SHUTDOWN("shutdown");

  companion object {
    fun fromCode(code: Int): AndroidThermalStatus? =
      when (code) {
        0 -> NONE
        1 -> LIGHT
        2 -> MODERATE
        3 -> SEVERE
        4 -> CRITICAL
        5 -> EMERGENCY
        6 -> SHUTDOWN
        else -> null
      }
  }
}

fun androidThermalStatusFromCode(code: Int): AndroidThermalStatus? = AndroidThermalStatus.fromCode(code)

fun parseAndroidThermalStatus(stdout: String): AndroidThermalStatus? {
  for (line in stdout.lineSequence()) {
    val trimmed = line.trim()
    if (trimmed.startsWith("Thermal Status:")) {
      val rest = trimmed.substring("Thermal Status:".length).trim()
      val code = rest.toIntOrNull() ?: return null
      return androidThermalStatusFromCode(code)
    }
  }
  return null
}

fun parseAndroidHeadroom(stdout: String): Float? {
  for (line in stdout.lineSequence()) {
    if (line.contains("Headroom")) {
      val idx = line.lastIndexOf(':')
      if (idx >= 0) {
        val value = line.substring(idx + 1).trim().toFloatOrNull()
        if (value != null && value.isFinite()) {
          return value
        }
      }
    }
  }
  return null
}

data class AndroidTemperatureSensor(val iteration: Int, val sensorType: String, val name: String, val celsius: Int, val throttlingStatus: String)

fun parseAndroidHalSensors(stdout: String): List<AndroidTemperatureSensor>? {
  var inCurrent = false
  val sensors = mutableListOf<AndroidTemperatureSensor>()

  for (line in stdout.lineSequence()) {
    val trimmed = line.trim()
    if (trimmed == "Current temperatures from HAL:") {
      inCurrent = true
      continue
    }
    if (!inCurrent) continue
    if (!trimmed.startsWith("Temperature{")) {
      inCurrent = false
      continue
    }

    val mType = dumpsysField(trimmed, "mType=")?.toIntOrNull() ?: continue
    val sensorType = androidTemperatureType(mType) ?: continue
    val value = dumpsysField(trimmed, "mValue=")?.toDoubleOrNull() ?: continue
    if (!value.isFinite() || value < -50.0 || value > 200.0) {
      continue
    }
    val mStatus = dumpsysField(trimmed, "mStatus=")?.toIntOrNull() ?: continue
    val name = dumpsysField(trimmed, "mName=") ?: continue

    sensors.add(
      AndroidTemperatureSensor(
        iteration = 0,
        sensorType = sensorType,
        name = name,
        celsius = value.roundToInt(),
        throttlingStatus = androidThrottlingSeverity(mStatus),
      )
    )
  }

  return if (sensors.isNotEmpty()) sensors else null
}

private fun androidTemperatureType(mType: Int): String? =
  when (mType) {
    0 -> "cpu"
    1 -> "gpu"
    2 -> "battery"
    3 -> "skin"
    4 -> "usb_port"
    5 -> "power_amplifier"
    9 -> "npu"
    10 -> "tpu"
    11 -> "display"
    12 -> "modem"
    13 -> "soc"
    14 -> "wifi"
    15 -> "camera"
    16 -> "flashlight"
    17 -> "speaker"
    18 -> "ambient"
    19 -> "pogo"
    else -> null
  }

private fun androidThrottlingSeverity(mStatus: Int): String =
  when (mStatus) {
    1 -> "light"
    2 -> "moderate"
    3 -> "severe"
    4 -> "critical"
    5 -> "emergency"
    6 -> "shutdown"
    else -> "none"
  }

private fun dumpsysField(line: String, key: String): String? {
  val start = line.indexOf(key)
  if (start < 0) return null
  val valueStart = start + key.length
  val rest = line.substring(valueStart)
  val endComma = rest.indexOf(',')
  val endBrace = rest.indexOf('}')
  val end =
    when {
      endComma >= 0 && endBrace >= 0 -> kotlin.math.min(endComma, endBrace)
      endComma >= 0 -> endComma
      endBrace >= 0 -> endBrace
      else -> rest.length
    }
  return rest.substring(0, end).trim()
}
