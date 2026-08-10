package ai.liquid.pipette

import android.app.AlertDialog
import android.text.InputType
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import io.sentry.Sentry
import io.sentry.protocol.Feedback

/**
 * In-app user feedback / bug reporting, recorded as a Sentry User Feedback record via the [Sentry.feedback] API. This mirrors the pipette-dashboard
 * web flow (dialog → Sentry.feedback().capture()) so submissions land in Sentry with the same shape and tag taxonomy — message + optional email in
 * the native feedback fields, and category / app version / device as filterable tags.
 *
 * Differences from the web flow, by design:
 * - No honeypot / min-submit-interval spam guard. Those defend against bots crawling a public DOM and auto-filling a form; a native Material dialog
 *   has no such surface, so the guard would only add friction for real users.
 * - Tags carry device/chip/OS (useful for an on-device benchmarking client) instead of the browser's viewport size.
 *
 * The feature gates on [Sentry.isEnabled] — the same idea as the web's FEEDBACK_ENABLED (which keys off the DSN). If Sentry didn't initialize, the
 * Settings entry is hidden.
 */
object FeedbackDialog {
  /** Source tag value marking where a submission originated (web uses "feedback-dialog"). */
  private const val SOURCE = "android-settings"

  const val BUTTON_LABEL = "Submit feedback"
  const val DIALOG_TITLE = "Submit feedback"
  const val DIALOG_DESCRIPTION = "Tell us what's missing or broken. We read every message."

  // Dialog layout spacing (dp), named so they read as design intent (and to satisfy
  // detekt's MagicNumber rule — the kit's own dp() call sites are baselined).
  private const val CONTENT_SIDE_PAD_DP = 20
  private const val CONTENT_TOP_PAD_DP = 8
  private const val FIELD_LABEL_TOP_PAD_DP = 10
  private const val FIELD_LABEL_BOTTOM_PAD_DP = 2
  private const val MESSAGE_MIN_LINES = 4

  /**
   * Optional category. Kept in sync with pipette-dashboard's FEEDBACK_CATEGORIES so the `category` tag means the same thing across web and Android in
   * Sentry.
   */
  private data class Category(val id: String, val label: String)

  private val CATEGORIES =
    listOf(
      Category("report_bug", "Report a bug"),
      Category("report_incorrect_data", "Report incorrect data"),
      Category("request_model", "Request a model"),
      Category("request_runtime", "Request a runtime"),
      Category("request_hardware", "Request hardware"),
      Category("request_eval", "Request an evaluation dataset"),
      Category("other", "Something else"),
    )

  /**
   * The category ids, in submission order. Exposed for the cross-platform parity test ([FeedbackCategoryTest]) that pins them against the iOS
   * `FeedbackCategory` / web list so a rename or reorder on one platform can't silently change what the `category` tag means.
   */
  internal val CATEGORY_IDS: List<String>
    get() = CATEGORIES.map { it.id }

  /** True when Sentry initialized (DSN wired via the manifest). Callers hide the entry otherwise. */
  fun isAvailable(): Boolean = Sentry.isEnabled()

  /** (id, label) category options for a picker, in the pinned [CATEGORY_IDS] order. */
  val options: List<Pair<String, String>>
    get() = CATEGORIES.map { it.id to it.label }

  /**
   * Record a feedback submission from any UI (the Compose Settings screen or the legacy View dialog). [categoryId] is one of [CATEGORY_IDS] or null
   * (no category). Fire-and-forget via Sentry.
   */
  fun capture(message: String, email: String, categoryId: String?, analytics: Analytics = NoOpAnalytics) {
    submit(message, email, categoryId?.let { id -> CATEGORIES.firstOrNull { it.id == id } }, analytics)
  }

  /**
   * Show the feedback dialog. [defaultEmail] pre-fills the optional reply address (the signed-in Clerk email, when available). [onSubmitted] runs
   * after a successful capture so the caller can surface a status line / re-render.
   */
  fun show(activity: AppCompatActivity, ui: UiKit, defaultEmail: String?, analytics: Analytics = NoOpAnalytics, onSubmitted: () -> Unit) {
    // null = no category chosen (the optional default), matching the web's empty select.
    var selectedCategory: Category? = null

    val categoryButton = ui.outlineButton(categoryLabel(null)) {}
    categoryButton.setOnClickListener {
      // Single-choice picker; index 0 clears the selection back to "optional".
      val labels = (listOf("None (optional)") + CATEGORIES.map { it.label }).toTypedArray()
      val checked = selectedCategory?.let { CATEGORIES.indexOf(it) + 1 } ?: 0
      AlertDialog.Builder(activity)
        .setTitle("What's this about?")
        .setSingleChoiceItems(labels, checked) { picker, which ->
          selectedCategory = if (which == 0) null else CATEGORIES[which - 1]
          // Reflect the choice on the button face (it otherwise stays "Category: optional").
          categoryButton.text = categoryLabel(selectedCategory)
          picker.dismiss()
        }
        .show()
    }

    val message =
      ui.input("What would you like to tell us?", "").apply {
        // Multiline: the kit's input() is single-line by default. Reuse its styled box but
        // let the message grow.
        setSingleLine(false)
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        minLines = MESSAGE_MIN_LINES
        gravity = Gravity.TOP or Gravity.START
      }

    val email = ui.input("you@example.com (optional)", defaultEmail ?: "", InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS or InputType.TYPE_CLASS_TEXT)

    val content =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(ui.dp(CONTENT_SIDE_PAD_DP), ui.dp(CONTENT_TOP_PAD_DP), ui.dp(CONTENT_SIDE_PAD_DP), 0)
        addView(ui.mutedLabel(DIALOG_DESCRIPTION))
        addView(fieldLabel(ui, "What's this about? (optional)"))
        addView(categoryButton)
        addView(fieldLabel(ui, "Tell us more *"))
        addView(message)
        addView(fieldLabel(ui, "Your email (optional, if you want a reply)"))
        addView(email)
      }

    val dialog =
      AlertDialog.Builder(activity)
        .setTitle(DIALOG_TITLE)
        .setView(content)
        .setPositiveButton("Submit", null) // overridden below so a blank message keeps the dialog open
        .setNegativeButton("Cancel", null)
        .create()

    dialog.setOnShowListener {
      dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
        val text = message.text.toString().trim()
        if (text.isEmpty()) {
          message.error = "Required"
          return@setOnClickListener
        }
        submit(text, email.text.toString().trim(), selectedCategory, analytics)
        dialog.dismiss()
        Toast.makeText(activity, "Thanks, we got your feedback.", Toast.LENGTH_LONG).show()
        onSubmitted()
      }
    }
    dialog.show()
  }

  /** Button face text reflecting the current category choice (defaults to "optional"). */
  private fun categoryLabel(category: Category?): String = "Category: ${category?.label ?: "optional"}"

  /** A small dimmed caption above each field, matching the kit's muted-label sizing. */
  private fun fieldLabel(ui: UiKit, text: String): TextView =
    ui.mutedLabel(text).apply { setPadding(0, ui.dp(FIELD_LABEL_TOP_PAD_DP), 0, ui.dp(FIELD_LABEL_BOTTOM_PAD_DP)) }

  /**
   * Hand off to Sentry. Message + email go into the native feedback fields; category, app version, source, and device info become tags (filterable in
   * the Sentry UI). The call is fire-and-forget — the SDK queues and ships the envelope on its own thread.
   *
   * Tags are attached via [Sentry.withScope] (which clones the scope for the block, so the tags scope to this one feedback) rather than the
   * deprecated ScopeCallback overload of capture().
   */
  private fun submit(message: String, email: String, category: Category?, analytics: Analytics) {
    val feedback =
      Feedback(message).apply {
        if (email.isNotEmpty()) contactEmail = email
        url = "pipette-android://settings/feedback"
      }
    Sentry.withScope { scope ->
      scope.setTag("source", SOURCE)
      scope.setTag("app_version", BuildConfig.VERSION_NAME)
      category?.let { scope.setTag("category", it.id) }
      scope.setTag("device", DeviceInfo.modelName())
      scope.setTag("chip", DeviceInfo.chipModel())
      scope.setTag("os", "Android ${DeviceInfo.osVersion()}")
      Sentry.feedback().capture(feedback)
    }
    // The feedback CONTENT stays in Sentry. PostHog records only that feedback was sent and
    // under which category, so the funnel can show how often people reach for it. No message,
    // no contact email.
    analytics.capture(AnalyticsEvents.FEEDBACK_SUBMITTED, mapOf(AnalyticsEvents.CATEGORY to category?.id))
  }
}
