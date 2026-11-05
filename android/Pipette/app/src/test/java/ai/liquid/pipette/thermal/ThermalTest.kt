package ai.liquid.pipette.thermal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ThermalTest {

  @Test
  fun testAndroidThermalStatusFromCode() {
    assertEquals(AndroidThermalStatus.NONE, androidThermalStatusFromCode(0))
    assertEquals(AndroidThermalStatus.LIGHT, androidThermalStatusFromCode(1))
    assertEquals(AndroidThermalStatus.MODERATE, androidThermalStatusFromCode(2))
    assertEquals(AndroidThermalStatus.SEVERE, androidThermalStatusFromCode(3))
    assertEquals(AndroidThermalStatus.CRITICAL, androidThermalStatusFromCode(4))
    assertEquals(AndroidThermalStatus.EMERGENCY, androidThermalStatusFromCode(5))
    assertEquals(AndroidThermalStatus.SHUTDOWN, androidThermalStatusFromCode(6))
    assertNull(androidThermalStatusFromCode(-1))
    assertNull(androidThermalStatusFromCode(7))
  }

  @Test
  fun testParseAndroidThermalStatus() {
    val output =
      """
      DUMP OF SERVICE thermalservice:
      Is HAL ready: true
      HAL connection:
        Thermal HAL 2.0
      Thermal Status: 2
      Cached temperatures:
      """
        .trimIndent()

    assertEquals(AndroidThermalStatus.MODERATE, parseAndroidThermalStatus(output))
    assertNull(parseAndroidThermalStatus("no status here"))
    assertNull(parseAndroidThermalStatus("Thermal Status: not_a_number"))
  }

  @Test
  fun testParseAndroidHeadroom() {
    val output = "Headroom in 0 seconds: 0.72"
    val headroom = parseAndroidHeadroom(output)
    assertNotNull(headroom)
    assertEquals(0.72f, headroom!!, 0.001f)

    assertNull(parseAndroidHeadroom("Headroom in 0 seconds: NaN"))
    assertNull(parseAndroidHeadroom("No headroom here"))
  }

  @Test
  fun testParseAndroidHalSensors() {
    val dumpsys =
      """
      Cached temperatures:
        Temperature{mValue=50.0, mType=0, mName=cached-cpu, mStatus=0}
      Current temperatures from HAL:
        Temperature{mValue=32.4, mType=0, mName=cpu-0, mStatus=0}
        Temperature{mValue=45.6, mType=1, mName=gpu, mStatus=1}
        Temperature{mValue=-3.4028235E38, mType=2, mName=bad-batt, mStatus=0}
        Temperature{mValue=30.0, mType=6, mName=bcl-voltage, mStatus=0}
        Temperature{mValue=28.1, mType=13, mName=soc, mStatus=2}
      Some other section:
        Temperature{mValue=99.0, mType=0, mName=after-hal, mStatus=0}
      """
        .trimIndent()

    val sensors = parseAndroidHalSensors(dumpsys)
    assertNotNull(sensors)
    assertEquals(3, sensors!!.size)

    assertEquals("cpu", sensors[0].sensorType)
    assertEquals("cpu-0", sensors[0].name)
    assertEquals(32, sensors[0].celsius)
    assertEquals("none", sensors[0].throttlingStatus)

    assertEquals("gpu", sensors[1].sensorType)
    assertEquals("gpu", sensors[1].name)
    assertEquals(46, sensors[1].celsius)
    assertEquals("light", sensors[1].throttlingStatus)

    assertEquals("soc", sensors[2].sensorType)
    assertEquals("soc", sensors[2].name)
    assertEquals(28, sensors[2].celsius)
    assertEquals("moderate", sensors[2].throttlingStatus)
  }
}
