import Foundation

/// Raises iOS's Local Network alert, which `cdn-proxy=` downloads need answered. iOS has no
/// API for it; Apple's TN3179 connects a UDP socket to a link-local address, which raises it
/// without sending anything. Best effort.
nonisolated enum LocalNetworkAlert {
    static func trigger() {
        for address in linkLocalPeers() {
            let sock = socket(AF_INET6, SOCK_DGRAM, 0)
            guard sock >= 0 else { return }
            defer { close(sock) }
            withUnsafePointer(to: address) { sa6 in
                sa6.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    _ = connect(sock, sa, socklen_t(sa.pointee.sa_len))
                }
            }
        }
    }

    /// On the local network, but no particular device: random host parts, the discard port.
    private static func linkLocalPeers() -> [sockaddr_in6] {
        let hostParts = (0..<2).map { _ in (0..<8).map { _ in UInt8.random(in: 0...255) } }
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let start = list else { return [] }
        defer { freeifaddrs(start) }
        return sequence(first: start, next: { $0.pointee.ifa_next })
            .compactMap { ifa -> sockaddr_in6? in
                guard ifa.pointee.ifa_flags & UInt32(bitPattern: IFF_BROADCAST) != 0,
                      let sa = ifa.pointee.ifa_addr,
                      sa.pointee.sa_family == AF_INET6,
                      sa.pointee.sa_len >= MemoryLayout<sockaddr_in6>.size
                else { return nil }
                return UnsafeRawPointer(sa).load(as: sockaddr_in6.self)
            }
            .filter { $0.sin6_addr.__u6_addr.__u6_addr8.0 == 0xfe
                && ($0.sin6_addr.__u6_addr.__u6_addr8.1 & 0xc0) == 0x80 }
            .flatMap { address in
                hostParts.map { hostPart in
                    var peer = address
                    peer.sin6_port = UInt16(9).bigEndian
                    withUnsafeMutableBytes(of: &peer.sin6_addr) {
                        $0[8...].copyBytes(from: hostPart)
                    }
                    return peer
                }
            }
    }
}
