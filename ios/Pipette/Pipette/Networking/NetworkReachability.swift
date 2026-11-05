import Foundation
import Network

/// The reachability `DownloadCoordinator` reads, behind a seam so a test fixes the answer
/// rather than depending on the machine and on a monitor only the app's launch starts.
protocol NetworkReachable: Sendable {
    nonisolated var isConnected: Bool { get }

    /// Separates a reported `isConnected == false` from the pessimistic default an unwarmed
    /// monitor answers with. A caller that treats the two alike declines online work on an
    /// online device.
    nonisolated var hasReading: Bool { get }
}

/// Always-on network reachability, backed by one shared `NWPathMonitor`.
///
/// `nonisolated` (not the project-default `@MainActor`) and lock-guarded so the
/// off-main benchmark run loop can read `isConnected` synchronously.
nonisolated final class NetworkReachability: @unchecked Sendable {
    static let shared = NetworkReachability()

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "ai.liquid.pipette.reachability", qos: .utility)
    private let lock = NSLock()
    private var satisfied = false
    private var receivedReading = false
    private var started = false
    private var reconnectHandler: (@Sendable () -> Void)?

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            let now = path.status == .satisfied
            self.lock.lock()
            // Not on the first reading: the pessimistic `false` default is not
            // a real "was offline" state.
            let reconnected = self.receivedReading && !self.satisfied && now
            self.satisfied = now
            self.receivedReading = true
            let handler = self.reconnectHandler
            self.lock.unlock()
            if reconnected { handler?() }
        }
    }

    /// Idempotent. Call at launch so `isConnected` is accurate before the first
    /// benchmark cell completes.
    func start() {
        lock.lock()
        let alreadyStarted = started
        started = true
        lock.unlock()
        guard !alreadyStarted else { return }
        monitor.start(queue: queue)
    }

    /// Fired on an unsatisfied-to-satisfied transition, not for the initial
    /// reading. One handler: a later registration replaces the previous one.
    func onReconnect(_ handler: @escaping @Sendable () -> Void) {
        lock.lock()
        reconnectHandler = handler
        lock.unlock()
    }

    /// Pessimistic until the monitor first reports a satisfied path. The monitor
    /// reports within milliseconds of `start`, long before the first cell finishes.
    /// That default suits a caller who asks again later (the per-cell upload gate);
    /// a caller who acts on the answer once reads `hasReading` first.
    var isConnected: Bool {
        lock.lock()
        defer { lock.unlock() }
        return satisfied
    }

    var hasReading: Bool {
        lock.lock()
        defer { lock.unlock() }
        return receivedReading
    }
}

extension NetworkReachability: NetworkReachable {}
