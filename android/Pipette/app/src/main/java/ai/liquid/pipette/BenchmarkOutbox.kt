package ai.liquid.pipette

import io.sentry.Attachment
import io.sentry.ScopesAdapter
import io.sentry.Sentry
import io.sentry.SentryEnvelope
import io.sentry.SentryEnvelopeItem
import io.sentry.SentryEvent
import java.io.File
import java.util.UUID

/**
 * Writes a reconstructed `:benchmark` event (native-crash reconstruction, JVM crash, or OOM/ANR) into the Sentry SDK **outbox** as a serialized
 * envelope, instead of `Sentry.captureEvent`.
 *
 * These events describe the ISOLATED `:benchmark` process, not the UI process. `Sentry.captureEvent` would run them through the main hub's session
 * bookkeeping, so an unhandled-crash event flips the MAIN process's release-health session to Crashed — dragging the app's crash-free-sessions rate
 * down with subprocess deaths that have nothing to do with UI health. Dropping the envelope straight into the outbox delivers it via the SDK (with
 * its retry / rate-limit / offline caching) WITHOUT touching the session.
 *
 * Delivery timing: the outbox is swept by `SendCachedEnvelopeIntegration` at the next app launch (and on network reconnect). The live outbox
 * `FileObserver` only fires on `CLOSE_WRITE`, i.e. an in-place write, so an envelope RENAMED into the outbox (as here — a rename is the only way to
 * land it atomically, without the sweep ever seeing a partial file) waits for that sweep rather than sending the instant it arrives. That's the
 * standard "reported on next launch" model for crash reporting and is acceptable here.
 */
internal object BenchmarkOutbox {
  /**
   * Serialize [event] (plus any [attachments]) into the outbox atomically (temp + rename). Returns false when Sentry is disabled or has no outbox (no
   * DSN configured) so the caller can leave the source file for a later retry.
   */
  fun writeEvent(event: SentryEvent, attachments: List<Attachment> = emptyList()): Boolean {
    // No outbox unless the JVM SDK is initialized with a DSN (skips local/no-Sentry builds). We also need the outbox's PARENT dir up front: the temp
    // file is written there (see below), so bail now if it's somehow absent. Folded into one guard to stay within detekt's ReturnCount.
    val options = if (Sentry.isEnabled()) ScopesAdapter.getInstance().options else null
    val outboxPath = options?.outboxPath?.takeIf { it.isNotBlank() }
    val parent = outboxPath?.let { File(it).parentFile }
    if (options == null || outboxPath == null || parent == null) return false
    val serializer = options.serializer

    // These events describe `:benchmark` but are delivered from the main process via the outbox, whose sweeper re-captures them with a Cached hint
    // (which suppresses scope application). So stamp release/environment/dist from options here to guarantee correct release attribution. The native
    // `.envelope` path already embeds these at sentry_init; release builds aren't minified, so no ProGuard debug image is needed.
    if (event.release == null) event.release = options.release
    if (event.environment == null) event.environment = options.environment
    if (event.dist == null) event.dist = options.dist

    val outboxDir = File(outboxPath).apply { mkdirs() }
    val dest = File(outboxDir, "${UUID.randomUUID()}.envelope")
    // The temp file goes in the outbox's PARENT (resolved above), NOT the outbox: the outbox FileObserver reacts to any file appearing in the dir
    // (including a ".tmp") and would try — and fail — to send the half-written/renamed-away partial. The observer must only ever see the finished
    // envelope arrive via the atomic rename (MOVED_TO), same as the native path. Parent is same-filesystem, so the rename stays atomic.
    val tmp = File(parent, "${dest.name}.tmp")
    // Everything that can throw (envelope-item construction via fromEvent/fromAttachment, plus serialization and rename) stays inside runCatching,
    // so this function honors its Boolean contract and never throws into the caller.
    return runCatching {
        val items = ArrayList<SentryEnvelopeItem>(1 + attachments.size)
        items.add(SentryEnvelopeItem.fromEvent(serializer, event))
        for (attachment in attachments) {
          items.add(SentryEnvelopeItem.fromAttachment(serializer, options.logger, attachment, options.maxAttachmentSize))
        }
        val envelope = SentryEnvelope(event.eventId, options.sdkVersion, items)
        tmp.outputStream().use { serializer.serialize(envelope, it) }
        tmp.renameTo(dest) ||
          run {
            tmp.delete()
            false
          }
      }
      .getOrElse {
        tmp.delete()
        false
      }
  }
}
