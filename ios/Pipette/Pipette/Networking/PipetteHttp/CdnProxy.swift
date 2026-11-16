import Foundation
import Network

/// A caching proxy on the local network: the `pipette-http` crate's `CdnProxy`, the CLI's
/// `--cdn-proxy`. A launch names it with `cdn-proxy=HOST:PORT`, and every model download goes
/// to it as the request a direct fetch would send, `http://HOST:PORT<path>?<query>` with
/// `Host` naming the site, over plain HTTP and with a repo's token. Nothing falls back to the
/// site.
nonisolated struct CdnProxy: Sendable, Equatable {
    /// `http://HOST:PORT`.
    let baseURL: String

    /// The crate's `FromStr`: no scheme or path, a port from 1 to 65535, and a name, an IPv4
    /// address or an IPv6 address in brackets.
    init?(_ value: String) {
        guard !value.contains("://"), let colon = value.lastIndex(of: ":"),
              let port = UInt16(value[value.index(after: colon)...]), port != 0
        else { return nil }
        let host = value[..<colon]
        let valid = if host.hasPrefix("["), host.hasSuffix("]") {
            IPv6Address(String(host.dropFirst().dropLast())) != nil
        } else {
            !host.isEmpty
                && host.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "._-".contains($0)) }
        }
        guard valid else { return nil }
        baseURL = "http://\(host):\(port)"
    }

    /// The spelling `pipette-ios-proxy` passes.
    static let argument = "cdn-proxy="

    static func value(in arguments: [String]) -> String? {
        arguments.last { $0.hasPrefix(argument) }.map { String($0.dropFirst(argument.count)) }
    }

    /// A launch setting, as `-v` is: `HeadlessCommand.parse` drops the token before the
    /// grammar sees it and refuses a value this cannot read.
    static let launch: CdnProxy? = value(in: CommandLine.arguments).flatMap(CdnProxy.init)

    /// `request` as the proxy takes it, its other headers kept, a token included.
    func route(_ request: URLRequest) -> URLRequest {
        guard let url = request.url,
              let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let site = parts.encodedHost, !site.isEmpty,
              let routed = URL(string: baseURL + parts.percentEncodedPath
                                + (parts.percentEncodedQuery.map { "?\($0)" } ?? ""))
        // A `file://` source names no site and moves no byte over the network.
        else { return request }
        var proxied = request
        proxied.url = routed
        proxied.setValue(parts.port.map { "\(site):\($0)" } ?? site, forHTTPHeaderField: "Host")
        return proxied
    }
}
