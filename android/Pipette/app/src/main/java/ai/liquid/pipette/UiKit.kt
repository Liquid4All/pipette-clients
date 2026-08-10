package ai.liquid.pipette

import android.app.AlertDialog
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.core.graphics.ColorUtils
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.progressindicator.LinearProgressIndicator

/**
 * Programmatic view factory + small dialog/toast helpers shared by every screen.
 *
 * This is the app's design system. Headers use the built-in `serif` family (Noto Serif on Android) to echo the iOS client's serif titles; body text
 * stays on the platform sans (Roboto). Colors come from the dark-only palette in colors.xml. The richer builders (`card`, `primaryButton`, `chip`,
 * `statusBadge`, `searchField`, `linearProgress`, `segmented`) wrap Material Components so the screens can drop their raw
 * `TextView`/`EditText`/`Button` chrome.
 */
class UiKit(private val context: Context) {
  private val serif: Typeface = Typeface.create("serif", Typeface.NORMAL)
  private val serifBold: Typeface = Typeface.create("serif", Typeface.BOLD)

  fun dp(value: Int): Int = value.dp

  private fun color(resId: Int): Int = ContextCompat.getColor(context, resId)

  // --- Palette accessors (so screens read semantic names, not raw resources) ---
  fun colorPrimary(): Int = color(R.color.pipette_primary)

  fun colorOnPrimary(): Int = color(R.color.pipette_on_primary)

  fun colorSurface(): Int = color(R.color.pipette_surface)

  fun colorOnSurface(): Int = color(R.color.pipette_on_surface)

  fun colorMuted(): Int = color(R.color.pipette_on_surface_muted)

  fun colorOutline(): Int = color(R.color.pipette_outline)

  // --- Typography -------------------------------------------------------------

  /** Large serif page title (iOS uses IowanOldStyle here). */
  fun displayTitle(textValue: String): TextView =
    TextView(context).apply {
      text = textValue
      textSize = 26f
      typeface = serifBold
      setTextColor(colorOnSurface())
    }

  fun sectionTitle(textValue: String): TextView =
    TextView(context).apply {
      text = textValue
      textSize = 18f
      typeface = serifBold
      setTextColor(colorOnSurface())
      setPadding(0, dp(18), 0, dp(6))
    }

  fun label(textValue: String): TextView =
    TextView(context).apply {
      text = textValue
      textSize = 14f
      setTextColor(colorOnSurface())
      setPadding(0, dp(4), 0, dp(8))
    }

  /** Secondary/caption text — dimmed, for metadata and hints. */
  fun mutedLabel(textValue: String): TextView =
    TextView(context).apply {
      text = textValue
      textSize = 13f
      setTextColor(colorMuted())
      setPadding(0, dp(2), 0, dp(6))
    }

  // --- Containers -------------------------------------------------------------

  /**
   * A rounded surface card with a hairline outline (iOS `AppListCard`). The lambda configures the inner padded vertical column; the returned card
   * already carries its own vertical margin, so it can be added straight to a body column.
   */
  fun card(buildContent: LinearLayout.() -> Unit): MaterialCardView {
    val column =
      LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), dp(14), dp(16), dp(16))
        buildContent()
      }
    return MaterialCardView(context).apply {
      radius = dp(18).toFloat()
      cardElevation = 0f
      strokeWidth = dp(1)
      setStrokeColor(colorOutline())
      setCardBackgroundColor(colorSurface())
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(8), 0, dp(8)) }
      addView(column)
    }
  }

  /**
   * A lighter rounded "tile" for list rows that live *inside* a [card] (model rows, download rows, job/cell cards) — uses the surface-variant fill so
   * it reads as a nested item without the cost of a full MaterialCardView per row.
   */
  fun tile(buildContent: LinearLayout.() -> Unit): LinearLayout =
    LinearLayout(context).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(12), dp(10), dp(12), dp(12))
      background =
        GradientDrawable().apply {
          cornerRadius = dp(12).toFloat()
          setColor(color(R.color.pipette_surface_variant))
        }
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(6), 0, 0) }
      buildContent()
    }

  // --- Buttons ----------------------------------------------------------------

  /** Prominent filled action (iOS black capsule). */
  fun primaryButton(textValue: String, onClick: () -> Unit): MaterialButton =
    MaterialButton(context).apply {
      text = textValue
      isAllCaps = false
      cornerRadius = dp(14)
      setOnClickListener { onClick() }
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(6), 0, dp(6)) }
    }

  /** Secondary outlined action. */
  fun outlineButton(textValue: String, onClick: () -> Unit): MaterialButton =
    MaterialButton(context, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
      text = textValue
      isAllCaps = false
      cornerRadius = dp(14)
      setOnClickListener { onClick() }
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(6), 0, dp(6)) }
    }

  /**
   * Low-emphasis text-only action (inline, e.g. row affordances). Built flat by hand: a default [MaterialButton] is a *filled* button, and the
   * text-button style doesn't apply reliably via a programmatic ContextThemeWrapper (it fell back to filled — white text on a white fill).
   * Transparent background tint + primary-colored text + no stroke/elevation gives the text-button look.
   */
  fun textButton(textValue: String, onClick: () -> Unit): MaterialButton =
    MaterialButton(context).apply {
      text = textValue
      isAllCaps = false
      backgroundTintList = ColorStateList.valueOf(Color.TRANSPARENT)
      setTextColor(colorPrimary())
      strokeWidth = 0
      elevation = 0f
      insetTop = 0
      insetBottom = 0
      // Compact, left-aligned link — not a full-width centered button. Default
      // MaterialButton width fills a vertical column and centers its label,
      // which makes inline row affordances (Details / Rerun / Apply) look
      // misaligned next to a left-aligned checkbox.
      minWidth = 0
      minimumWidth = 0
      setPadding(dp(8), dp(4), dp(8), dp(4))
      layoutParams = LinearLayout.LayoutParams(WRAP, WRAP)
      setOnClickListener { onClick() }
    }

  // --- Chips / badges ---------------------------------------------------------

  /** Selectable filter chip (iOS quant/benchmark pills). */
  fun filterChip(textValue: String, selected: Boolean, onToggle: (Boolean) -> Unit): Chip =
    Chip(context).apply {
      text = textValue
      isCheckable = true
      isChecked = selected
      isAllCaps = false
      setOnCheckedChangeListener { _, isChecked -> onToggle(isChecked) }
    }

  /** A wrapping container for chips (auto-flows to multiple lines). */
  fun chipGroup(build: ChipGroup.() -> Unit): ChipGroup =
    ChipGroup(context).apply {
      isSingleSelection = false
      build()
    }

  /** Static informational chip (no selection). */
  fun infoChip(textValue: String): Chip =
    Chip(context).apply {
      text = textValue
      isClickable = false
      isCheckable = false
    }

  /** Small rounded status pill with a tinted background (iOS `StatusBadge`). [accent] colors both the text and a low-alpha background fill. */
  fun statusBadge(textValue: String, accent: Int): TextView =
    TextView(context).apply {
      text = textValue
      textSize = 12f
      setTextColor(accent)
      setTypeface(typeface, Typeface.BOLD)
      setPadding(dp(10), dp(4), dp(10), dp(4))
      background =
        GradientDrawable().apply {
          cornerRadius = dp(999).toFloat()
          setColor(ColorUtils.setAlphaComponent(accent, 0x22))
        }
      // Hug the text so it reads as a pill, not a full-width bar, inside a
      // vertical column (whose default child width is MATCH_PARENT).
      layoutParams = LinearLayout.LayoutParams(WRAP, WRAP).apply { setMargins(0, dp(2), 0, dp(6)) }
    }

  /**
   * A small circular monogram (first letter of a model family / brand) used as a lightweight stand-in for the iOS client's brand logos in the model
   * list. Avoids shipping trademarked artwork while still giving each family row a leading glyph.
   */
  fun monogramAvatar(letter: String, size: Int = AVATAR_SIZE_DP): TextView =
    TextView(context).apply {
      text = letter.take(1).uppercase()
      textSize = AVATAR_TEXT_SP
      gravity = Gravity.CENTER
      setTextColor(colorOnSurface())
      setTypeface(serifBold)
      background =
        GradientDrawable().apply {
          shape = GradientDrawable.OVAL
          setColor(color(R.color.pipette_surface_variant))
          setStroke(1.dp, colorOutline())
        }
      layoutParams = LinearLayout.LayoutParams(size.dp, size.dp).apply { setMargins(0, 0, 12.dp, 0) }
    }

  fun colorThermalNominal(): Int = color(R.color.pipette_thermal_nominal)

  fun colorThermalSerious(): Int = color(R.color.pipette_thermal_serious)

  fun colorThermalCritical(): Int = color(R.color.pipette_thermal_critical)

  // --- Inputs -----------------------------------------------------------------

  /**
   * Every text field gets the same explicit dark rounded box + outline. The platform-default `EditText` background is NOT theme-safe — it renders
   * transparent on AOSP but the OEM default fills it light/white on some devices (e.g. Pixel), so fields looked inconsistent. Setting our own
   * background fixes the look everywhere and removes the reliance on the platform style.
   */
  fun input(hintValue: String, value: String, type: Int = InputType.TYPE_CLASS_TEXT): EditText =
    EditText(context).apply {
      hint = hintValue
      setText(value)
      inputType = type
      setSingleLine(true)
      setTextColor(colorOnSurface())
      setHintTextColor(colorMuted())
      background =
        GradientDrawable().apply {
          cornerRadius = dp(12).toFloat()
          setColor(color(R.color.pipette_surface_variant))
          setStroke(dp(1), colorOutline())
        }
      setPadding(dp(12), dp(12), dp(12), dp(12))
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(4), 0, dp(6)) }
    }

  /**
   * A search field. Now just a styled [input] (the box style is consistent with every other field); returns the same view as both halves of the pair
   * so callers can add it and read its text.
   */
  fun searchField(hintValue: String, value: String): Pair<View, EditText> {
    val field = input(hintValue, value)
    return field to field
  }

  // --- Progress ---------------------------------------------------------------

  /** Determinate slim progress bar (iOS capsule progress). [fraction] in 0..1. */
  fun linearProgress(fraction: Double): LinearProgressIndicator =
    LinearProgressIndicator(context).apply {
      max = 1000
      progress = (fraction.coerceIn(0.0, 1.0) * 1000).toInt()
      trackCornerRadius = dp(4)
      setIndicatorColor(colorPrimary())
      layoutParams = LinearLayout.LayoutParams(MATCH, WRAP).apply { setMargins(0, dp(8), 0, dp(8)) }
    }

  // --- Segmented control ------------------------------------------------------

  /** Single-select segmented control (iOS runtime picker). */
  fun segmented(options: List<String>, selectedIndex: Int, onSelect: (Int) -> Unit): MaterialButtonToggleGroup {
    val group =
      MaterialButtonToggleGroup(context).apply {
        isSingleSelection = true
        isSelectionRequired = true
      }
    options.forEachIndexed { index, optionLabel ->
      val button =
        MaterialButton(context, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
          text = optionLabel
          isAllCaps = false
          id = View.generateViewId()
          setOnClickListener { onSelect(index) }
        }
      group.addView(button, LinearLayout.LayoutParams(0, WRAP, 1f))
      if (index == selectedIndex) group.check(button.id)
    }
    return group
  }

  // --- Heatmap ----------------------------------------------------------------

  /**
   * Heatmap fill for a results cell. [intensity] in 0..1 is the value's rank within its column *after* direction normalization (so higher-is-better
   * and lower-is-better both map "better → brighter"). Higher intensity raises the alpha of the light primary color over the dark surface, so a
   * better cell reads brighter.
   */
  fun heatmapColor(intensity: Double): Int {
    val alpha = (0x14 + (intensity.coerceIn(0.0, 1.0) * (0x88 - 0x14))).toInt()
    return ColorUtils.setAlphaComponent(colorPrimary(), alpha)
  }

  // --- Legacy primitives (kept for screens not yet migrated to the kit) -------

  fun button(textValue: String, onClick: () -> Unit): Button =
    Button(context).apply {
      text = textValue
      isAllCaps = false
      setOnClickListener { onClick() }
      layoutParams = LinearLayout.LayoutParams(WRAP, WRAP).apply { setMargins(0, dp(4), dp(8), dp(4)) }
    }

  fun confirm(message: String, positiveText: String = "Delete", onConfirm: () -> Unit) {
    AlertDialog.Builder(context).setMessage(message).setPositiveButton(positiveText) { _, _ -> onConfirm() }.setNegativeButton("Cancel", null).show()
  }

  fun showError(error: Throwable) {
    Toast.makeText(context, error.message ?: error.javaClass.simpleName, Toast.LENGTH_LONG).show()
  }

  /** A thin horizontal row container (gravity-centered vertically). */
  fun row(gravity: Int = Gravity.CENTER_VERTICAL, build: LinearLayout.() -> Unit): LinearLayout =
    LinearLayout(context).apply {
      orientation = LinearLayout.HORIZONTAL
      this.gravity = gravity
      build()
    }

  /** Vendor brand logo for a model family (iOS `BrandLogoView`); falls back to a monogram when the vendor has no bundled logo. */
  fun brandLogo(name: String, hfRepo: String?, sizeDp: Int = AVATAR_SIZE_DP): View {
    val res = ModelBrand.logoRes(name, hfRepo)
    return if (res != null) {
      ImageView(context).apply {
        setImageResource(res)
        scaleType = ImageView.ScaleType.FIT_CENTER
        layoutParams = LinearLayout.LayoutParams(sizeDp.dp, sizeDp.dp).apply { setMargins(0, 0, 12.dp, 0) }
      }
    } else {
      monogramAvatar(name, sizeDp)
    }
  }

  /** Rounded checkbox matching the iOS `WizardCheckbox`: filled primary with a light check when on, an outlined empty square when off. */
  fun wizardCheckbox(checked: Boolean, sizeDp: Int = CHECKBOX_SIZE_DP): TextView =
    TextView(context).apply {
      text = if (checked) "✓" else ""
      textSize = CHECK_TEXT_SP
      gravity = Gravity.CENTER
      setTextColor(colorOnPrimary())
      typeface = Typeface.DEFAULT_BOLD
      background =
        GradientDrawable().apply {
          cornerRadius = 6.dp.toFloat()
          if (checked) {
            setColor(colorPrimary())
          } else {
            setColor(Color.TRANSPARENT)
            setStroke(2.dp, colorMuted())
          }
        }
      layoutParams = LinearLayout.LayoutParams(sizeDp.dp, sizeDp.dp)
    }

  /** Hairline horizontal divider used between list rows inside a card. */
  fun divider(): View =
    View(context).apply {
      setBackgroundColor(colorOutline())
      layoutParams = LinearLayout.LayoutParams(MATCH, 1.dp)
    }

  /**
   * Search field with a leading magnifier (iOS `AppSearchField`). Filters on the IME "search" action rather than per-keystroke: the screens rebuild
   * their whole view tree on every render, which would otherwise wipe focus mid-type. [onSubmit] receives the query and should re-render.
   */
  fun iconSearchField(hintValue: String, value: String, onSubmit: (String) -> Unit): EditText =
    input(hintValue, value).apply {
      val icon = ContextCompat.getDrawable(context, R.drawable.ic_search)?.mutate()
      icon?.setTint(colorMuted())
      icon?.setBounds(0, 0, 18.dp, 18.dp)
      // Relative (start) so the magnifier follows the text edge in RTL locales (the app declares supportsRtl).
      setCompoundDrawablesRelative(icon, null, null, null)
      compoundDrawablePadding = 10.dp
      imeOptions = EditorInfo.IME_ACTION_SEARCH
      setOnEditorActionListener { view, actionId, _ ->
        if (actionId == EditorInfo.IME_ACTION_SEARCH) {
          onSubmit(view.text.toString())
          true
        } else {
          false
        }
      }
    }

  /** Compact capsule action with an optional leading icon. [filled] = solid primary (iOS black pill); otherwise outlined. */
  fun pillButton(
    textValue: String,
    iconRes: Int? = null,
    filled: Boolean = false,
    fullWidth: Boolean = false,
    enabled: Boolean = true,
    onClick: () -> Unit,
  ): MaterialButton {
    val button =
      if (filled) MaterialButton(context) else MaterialButton(context, null, com.google.android.material.R.attr.materialButtonOutlinedStyle)
    return button.apply {
      text = textValue
      isAllCaps = false
      cornerRadius = CAPSULE_RADIUS_DP.dp
      isEnabled = enabled
      insetTop = 0
      insetBottom = 0
      if (!filled) setTextColor(colorPrimary())
      if (iconRes != null) {
        setIconResource(iconRes)
        iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
        iconPadding = 8.dp
        iconTint = ColorStateList.valueOf(if (filled) colorOnPrimary() else colorPrimary())
      }
      val width = if (fullWidth) MATCH else WRAP
      layoutParams = LinearLayout.LayoutParams(width, WRAP).apply { setMargins(0, 6.dp, 0, 6.dp) }
      setOnClickListener { onClick() }
    }
  }

  companion object {
    const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
    const val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
    private const val AVATAR_SIZE_DP = 34
    private const val AVATAR_TEXT_SP = 15f
    private const val CHECKBOX_SIZE_DP = 24
    private const val CHECK_TEXT_SP = 13f
    private const val CAPSULE_RADIUS_DP = 100
  }
}
