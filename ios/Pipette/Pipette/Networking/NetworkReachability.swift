import Foundation
import Network

/// Always-on network reachability, backed by one shared `NWPathMonitor`.
///
/// Two uses: a synchronous `isConnected` read so work that can only succeed
/// online (the per-cell result upload) skips itself when offline instead of
/// firing a doomed request; and an `onReconnect` callback so the download
/// coordinator can auto-resume transfers a connectivity drop interrupted.
///
/// `nonisolated` (not the project-default `@MainActor`) and lock-guarded so the
/// off-main benchmark run loop can read `isConnected` synchronously.
nonisolated final class NetworkReachability: @unchecked Sendable {
    static let shared = NetworkReachability()

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "ai.liquid.pipette.reachability", qos: .utility)
    private let lock = NSLock()
    private var satisfied = false
    private var hasReading = false
    private var started = false
    private var reconnectHandler: (@Sendable () -> Void)?

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            let now = path.status == .satisfied
            self.lock.lock()
            // Fire only on a genuine offline→online transition — never on the
            // first reading (the pessimistic `false` default isn't a real
            // "was offline" state).
            let reconnected = self.hasReading && !self.satisfied && now
            self.satisfied = now
            self.hasReading = true
            let handler = self.reconnectHandler
            self.lock.unlock()
            if reconnected { handler?() }
        }
    }

    /// Begin watching the network path. Idempotent; call once at launch so
    /// `isConnected` is accurate before the first benchmark cell completes.
    func start() {
        lock.lock()
        let alreadyStarted = started
        started = true
        lock.unlock()
        guard !alreadyStarted else { return }
        monitor.start(queue: queue)
    }

    /// Register a handler fired when connectivity returns (the path transitions
    /// from unsatisfied back to satisfied). Not fired for the initial reading.
    /// A single handler; a later registration replaces the previous one.
    func onReconnect(_ handler: @escaping @Sendable () -> Void) {
        lock.lock()
        reconnectHandler = handler
        lock.unlock()
    }

    /// Latest observed reachability. Thread-safe. Pessimistic until the monitor
    /// first reports a satisfied path — a not-yet-warmed monitor skips (and thus
    /// delays) an upload rather than firing a doomed offline request. The monitor
    /// delivers the current path within milliseconds of `start`, long before the
    /// first cell finishes, so an online run uploads from the first cell on.
    var isConnected: Bool {
        lock.lock()
        defer { lock.unlock() }
        return satisfied
    }
}
