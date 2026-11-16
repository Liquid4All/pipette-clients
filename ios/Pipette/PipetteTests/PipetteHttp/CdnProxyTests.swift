import Foundation
import Testing

@testable import Pipette

/// The value the crate's `CdnProxy` takes, the request the proxy receives, and how a launch
/// names it.
struct CdnProxyTests {
    private func proxy() throws -> CdnProxy { try #require(CdnProxy("127.0.0.1:8090")) }

    @Test(arguments: [
        ("[::1]:8090", "http://[::1]:8090"),
        ("cdn-cache.lan:8090", "http://cdn-cache.lan:8090"),
        ("192.168.1.20:8090", "http://192.168.1.20:8090"),
    ])
    func aValueIsAHostAndAPort(value: String, baseURL: String) {
        #expect(CdnProxy(value)?.baseURL == baseURL)
    }

    @Test(arguments: ["http://x:1", "x:1/models", ":8090", "x", "x:0", "x:65536", "::1:8090", "[x]:1"])
    func anythingElseIsRefused(value: String) {
        #expect(CdnProxy(value) == nil)
    }

    /// The request a direct fetch would send, byte for byte, so a signed query or an escaped
    /// segment reaches the site unchanged.
    @Test(arguments: [
        ("https://huggingface.co/org/repo/resolve/main/m.gguf",
         "http://127.0.0.1:8090/org/repo/resolve/main/m.gguf", "huggingface.co"),
        ("https://ex.com/a%2Fb/w.gguf?sig=x%3D1",
         "http://127.0.0.1:8090/a%2Fb/w.gguf?sig=x%3D1", "ex.com"),
        ("https://ex.com:8443/w.gguf", "http://127.0.0.1:8090/w.gguf", "ex.com:8443"),
    ])
    func aRequestGoesToTheProxyNamingItsSite(url: String, routed: String, site: String) throws {
        let sent = try proxy().route(URLRequest(url: try #require(URL(string: url))))
        #expect(sent.url?.absoluteString == routed)
        #expect(sent.value(forHTTPHeaderField: "Host") == site)
    }

    /// The site decides the token before the request is routed, so the Hub's reaches the
    /// proxy and no other site's request carries one.
    @Test func theHubsTokenGoesThroughTheProxy() throws {
        let token = try AuthToken("hf_fromtheclaim")
        let sent = try ["https://huggingface.co/org/repo/resolve/main/m.gguf", "https://ex.com/w.gguf"]
            .map { url in
                var request = URLRequest(url: try #require(URL(string: url)))
                attachHfAuth(&request, claimToken: token)
                return try proxy().route(request).value(forHTTPHeaderField: "Authorization")
            }
        #expect(sent == ["Bearer hf_fromtheclaim", nil])
    }

    @Test func aFileSourceIsNotRouted() throws {
        let request = URLRequest(url: try #require(URL(string: "file:///tmp/m.gguf")))
        #expect(try proxy().route(request) == request)
    }

    // MARK: - The launch argument

    @Test func theArgumentIsALaunchSettingThatNoVerbDeclares() {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        for argv in [["headlessrun", "version"],
                     ["headlessrun", "model=\(json)", "submit=1", "claim=job-1"]] {
            #expect(HeadlessCommand.parse(argv + ["cdn-proxy=127.0.0.1:8090"])
                == HeadlessCommand.parse(argv))
        }
    }

    @Test func aValueTheCrateRefusesIsRefused() {
        #expect(HeadlessCommand.parse(["headlessrun", "version", "cdn-proxy=http://x:1"])
            == .failure(.invalidValue(key: "cdn-proxy", value: "http://x:1")))
    }

    /// Any app can open a link, and the proxy receives the downloads' tokens.
    @Test func aLinkCannotNameAProxy() throws {
        let url = try #require(URL(string: "pipette://run?runtime=afm&cdn-proxy=127.0.0.1:8090"))
        #expect(HeadlessCommand.parse(url: url)
            == .failure(.rejected(key: "cdn-proxy", reason: "a link cannot name a proxy")))
    }
}
