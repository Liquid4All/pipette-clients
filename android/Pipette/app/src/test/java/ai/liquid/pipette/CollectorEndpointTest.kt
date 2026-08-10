package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Setup-screen collector choice: the URL a registration is aimed at. Mirrors iOS CollectorEndpointTests, because a self-hosted deployment configures
 * both clients with the same URL and they must agree on what that URL normalizes to.
 */
class CollectorEndpointTest {
  @Test
  fun endpointOptionsUseExpectedLabels() {
    assertEquals("Liquid AI", CollectorEndpointOption.PRODUCTION.title)
    assertEquals("Custom", CollectorEndpointOption.CUSTOM.title)
  }

  @Test
  fun productionOptionUsesHttpsCollectorUrl() {
    assertEquals("https://collector.pipette.liquid.ai", CollectorEndpointOption.PRODUCTION.serverUrl(""))
    // The DataStore default and the picker's production segment are the same endpoint; drift between them
    // would send a fresh install and a re-registration to different collectors.
    assertEquals(AppSettingsStore.DEFAULT_SERVER_URL, CollectorEndpointOption.PRODUCTION.serverUrl(""))
  }

  @Test
  fun customOptionAddsHttpsToBareHost() {
    assertEquals("https://collector.example.com", CollectorEndpointOption.CUSTOM.serverUrl("collector.example.com"))
  }

  @Test
  fun customOptionPreservesHttpsSchemeAndPathPrefix() {
    assertEquals("https://collector.example.com/pipette", CollectorEndpointOption.CUSTOM.serverUrl("https://collector.example.com/pipette/"))
  }

  @Test
  fun customOptionKeepsExplicitPort() {
    assertEquals("https://collector.example.com:8443", CollectorEndpointOption.CUSTOM.serverUrl("collector.example.com:8443"))
  }

  @Test
  fun customOptionLowercasesHost() {
    assertEquals("https://collector.example.com", CollectorEndpointOption.CUSTOM.serverUrl("https://Collector.Example.com"))
  }

  @Test
  fun customOptionRejectsHttpUnsupportedSchemesAndQueries() {
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("http://collector.example.com"))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("ftp://collector.example.com"))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("https://collector.example.com?token=1"))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("https://user:pw@collector.example.com"))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("https://collector.example.com#frag"))
  }

  @Test
  fun customOptionRejectsBlankAndUnparsableInput() {
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl(""))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("   "))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("collector example com"))
    assertNull(CollectorEndpointOption.CUSTOM.serverUrl("https://"))
  }
}
