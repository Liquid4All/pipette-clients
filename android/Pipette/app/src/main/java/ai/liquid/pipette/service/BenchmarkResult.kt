package ai.liquid.pipette.service

import android.os.Parcel
import android.os.Parcelable

/**
 * Outcome of a `:benchmark` engine operation, sent back across the AIDL boundary. On success [ok] is true and the payload is either [inlineJson]
 * (small results) or [referencePath] (a spilled file under the shared cache dir when the JSON would blow the Binder transaction limit); [handle] is
 * an opaque non-zero id for a successful load. On failure [ok] is false and [errorMessage] carries the native error the UI-process proxy re-throws.
 *
 * Manual [Parcelable] (no kotlin-parcelize plugin) to keep `:benchmark` and the build lean.
 */
data class BenchmarkResult(
  val ok: Boolean,
  val errorMessage: String? = null,
  val inlineJson: String? = null,
  val referencePath: String? = null,
  val handle: Long = 0L,
) : Parcelable {
  constructor(
    parcel: Parcel
  ) : this(
    ok = parcel.readInt() != 0,
    errorMessage = parcel.readString(),
    inlineJson = parcel.readString(),
    referencePath = parcel.readString(),
    handle = parcel.readLong(),
  )

  override fun writeToParcel(parcel: Parcel, flags: Int) {
    parcel.writeInt(if (ok) 1 else 0)
    parcel.writeString(errorMessage)
    parcel.writeString(inlineJson)
    parcel.writeString(referencePath)
    parcel.writeLong(handle)
  }

  override fun describeContents(): Int = 0

  companion object {
    @JvmField
    val CREATOR =
      object : Parcelable.Creator<BenchmarkResult> {
        override fun createFromParcel(parcel: Parcel): BenchmarkResult = BenchmarkResult(parcel)

        override fun newArray(size: Int): Array<BenchmarkResult?> = arrayOfNulls(size)
      }

    /** Largest JSON kept inline; anything bigger spills to a Reference file. */
    const val INLINE_LIMIT_BYTES = 256 * 1024

    fun failure(message: String): BenchmarkResult = BenchmarkResult(ok = false, errorMessage = message)

    fun ok(): BenchmarkResult = BenchmarkResult(ok = true)

    fun loaded(handle: Long): BenchmarkResult = BenchmarkResult(ok = true, handle = handle)
  }
}
