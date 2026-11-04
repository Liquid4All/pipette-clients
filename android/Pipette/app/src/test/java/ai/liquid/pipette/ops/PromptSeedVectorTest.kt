package ai.liquid.pipette.ops

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptSeedVectorTest {

  @Test
  fun testSeedTextResourceLoaded() {
    val text = PromptSeed.text
    assertNotNull("prompt_seed.txt must not be null", text)
    assertTrue("seed text must be substantial natural language", text.length > 20_000)
  }

  @Test
  fun testPromptSeedVectorsAgainstRustFixtures() {
    val stream = javaClass.classLoader?.getResourceAsStream("fixtures/prompt_seed_vectors.json")
    assertNotNull("prompt_seed_vectors.json fixture must be found", stream)
    val text = stream!!.bufferedReader().readText()
    val root = JSONObject(text)

    val promptCases = root.getJSONArray("prompt_seed_cases")
    for (i in 0 until promptCases.length()) {
      val obj = promptCases.getJSONObject(i)
      val tokenizer = obj.getString("tokenizer")
      val targetTokens = obj.getInt("target_tokens")
      val expectedText = obj.getString("expected_text")
      val tokenCount = obj.getInt("token_count")

      val countFn: (String) -> Int =
        when (tokenizer) {
          "byte_count" -> { s -> s.toByteArray(Charsets.UTF_8).size }
          "char_count" -> { s -> s.length }
          else -> error("Unknown tokenizer: $tokenizer")
        }

      val actualText = PromptSeed.buildPromptText(targetTokens, countFn)
      assertEquals("Text mismatch for target $targetTokens with $tokenizer", expectedText, actualText)
      assertEquals("Token count mismatch for target $targetTokens with $tokenizer", tokenCount, countFn(actualText))
    }

    val repeatCases = root.getJSONArray("repeat_token_cases")
    for (i in 0 until repeatCases.length()) {
      val obj = repeatCases.getJSONObject(i)
      val seedJson = obj.getJSONArray("seed")
      val seed = IntArray(seedJson.length()) { seedJson.getInt(it) }
      val targetLen = obj.getInt("target_len")
      val expectedJson = obj.getJSONArray("expected")
      val expected = IntArray(expectedJson.length()) { expectedJson.getInt(it) }

      val actual = PromptSeed.repeatTokenSequence(seed, targetLen)
      assertArrayEquals("repeatTokenSequence mismatch for targetLen $targetLen", expected, actual)
    }
  }
}
