// pipette-coreai-sidecar — HTTP server for pipette-coreai benchmarks.
//
// Serves the same JSON contract as the pipette-mlx Python sidecar
// (pipette_mlx_server.py), but over Apple's low-level Core AI runtime. It loads
// an .aimodel bundle directory (metadata.json + *.aimodel/ + tokenizer/) and
// exposes the benchmark endpoints the Rust client drives.
//
// Like Apple's own `llm-benchmark`, this uses the LOW-LEVEL Core AI API
// (LanguageBundle + EngineFactory + InferenceEngine) feeding RAW token IDs
// directly — no chat template. That is what lets pipette drive deterministic
// prompt/decode token counts on decode-only bundles, which the high-level
// LanguageModelSession path cannot do (it requires an embedded chat template).
//
// Endpoints (JSON, POST unless noted):
//   GET  /health                        -> {"ok": true}
//   POST /prefill_throughput            -> {"prompt_tps": f, "prompt_tokens": N}
//   POST /decode_throughput             -> {"generation_tps": f, "decode_tokens": N}
//   POST /max_memory_usage              -> {"prompt_tokens": N, "completion_tokens": M}
//   POST /end_to_end_latency            -> {"total_ms": f, "prompt_tokens": N, "completion_tokens": M}
//   POST /shutdown                      -> exits

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Network

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

struct Args {
    var modelDir: String
    var port: UInt16
    var seed: UInt64
}

enum SidecarError: Error, CustomStringConvertible {
    case missingArg(String)
    case unknownArg(String)
    case missingModel
    case badRequest(String)

    var description: String {
        switch self {
        case .missingArg(let a): return "missing value for argument \(a)"
        case .unknownArg(let a): return "unknown argument \(a)"
        case .missingModel: return "--model <bundle-dir> is required"
        case .badRequest(let m): return m
        }
    }
}

func parseArgs(_ args: [String]) throws -> Args {
    var modelDir: String?
    var port: UInt16 = 0
    var seed: UInt64 = 0
    var i = 0
    let argv = Array(args.dropFirst())
    func next() throws -> String {
        guard i + 1 < argv.count else { throw SidecarError.missingArg(argv[i]) }
        i += 1
        return argv[i]
    }
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--model": modelDir = try next()
        case "--port": port = UInt16(try next()) ?? 0
        case "--seed": seed = UInt64(try next()) ?? 0
        default: throw SidecarError.unknownArg(a)
        }
        i += 1
    }
    guard let modelDir else { throw SidecarError.missingModel }
    return Args(modelDir: modelDir, port: port, seed: seed)
}

// ---------------------------------------------------------------------------
// Low-level Core AI engine wrapper
// ---------------------------------------------------------------------------

final class CoreAIEngine: @unchecked Sendable {
    let engine: any InferenceEngine
    let vocabSize: Int
    let name: String
    let seed: UInt64

    init(bundleDir: URL, seed: UInt64) async throws {
        // S=1 zoo decode bundles (catalog hint "pipelined") are static graphs that
        // accept a single token per forward; multi-token prefill chunks are rejected
        // (shape-substitution fatal). COREAI_CHUNK_THRESHOLD=1 makes the runtime
        // chunk the prompt into 1-token steps — the same guard coreai-kit's
        // ModelRuntime sets for its .pipelined engine variant.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let bundle = try LanguageBundle(at: bundleDir)
        let modelURL = try bundle.requireModelURL(for: "main")
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let configData = try JSONEncoder().encode(config)
        self.engine = try await EngineFactory.createEngine(config: configData, modelURL: modelURL)
        self.vocabSize = bundle.vocabSize
        self.name = bundle.name
        self.seed = seed
    }

    /// Deterministic pseudo-random token prompt of exactly `count` tokens.
    func randomPrompt(count: Int) -> [Int32] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var out = [Int32]()
        out.reserveCapacity(count)
        let v = UInt64(vocabSize)
        for _ in 0..<count {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            out.append(Int32(z % v))
        }
        return out
    }

    /// Run one generation. Returns per-phase timing derived like Apple's
    /// llm-benchmark: promptTps over the prefill span (to first generated
    /// token), genTps over the decode span (from first token onward).
    func generate(prompt: [Int32], maxTokens: Int) async throws -> GenResult {
        // A short settle between generations. Without it the Core AI engine can
        // still be draining the previous decode when the next rep starts, which
        // perturbs the very first prompt_seconds measurement. Deliberately best-
        // effort (`try?`): a scheduling miss is not worth failing the run over.
        try? await Task.sleep(for: .milliseconds(50))
        try await engine.reset()

        let options = InferenceOptions(maxTokens: maxTokens, includeLogits: false)
        let sampling = SamplingConfiguration(temperature: 0)
        let start = ContinuousClock.now
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: sampling, inferenceOptions: options
        )

        var promptSeconds: Double = 0
        var genStart = ContinuousClock.now
        var count = 0
        var firstTokenAt: ContinuousClock.Instant?

        for try await _ in stream {
            if firstTokenAt == nil {
                firstTokenAt = ContinuousClock.now
                let now = ContinuousClock.now
                promptSeconds = seconds(from: start, to: now)
                genStart = now
            }
            count += 1
        }
        let genSeconds = seconds(from: genStart, to: .now)
        let promptTps = promptSeconds > 0 ? Double(prompt.count) / promptSeconds : 0
        let decodeCount = max(0, count - 1)
        let genTps = genSeconds > 0 ? Double(decodeCount) / genSeconds : 0
        return GenResult(
            promptTps: promptTps,
            genTps: genTps,
            promptTokens: prompt.count,
            completionTokens: decodeCount,
            genSeconds: genSeconds
        )
    }

    private func seconds(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: end)
        return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}

struct GenResult {
    let promptTps: Double
    let genTps: Double
    let promptTokens: Int
    let completionTokens: Int
    let genSeconds: Double
}

// ---------------------------------------------------------------------------
// Minimal HTTP server (Network framework)
// ---------------------------------------------------------------------------

private final class ReadBuffer: @unchecked Sendable {
    var data = Data()
}

final class SidecarServer: @unchecked Sendable {
    let port: UInt16
    let engine: CoreAIEngine
    let listener: NWListener

    init(port: UInt16, engine: CoreAIEngine) throws {
        self.port = port
        self.engine = engine
        guard let p = NWEndpoint.Port(rawValue: port) else {
            throw SidecarError.badRequest("invalid port")
        }
        // Restrict the listener to loopback so the benchmark endpoints
        // (including /shutdown) are never reachable from the LAN. The Rust
        // client always connects to 127.0.0.1, and the MLX sidecar binds the
        // same way. requiredInterfaceType must be set on the parameters before
        // the listener is created; NWListener.requiredLocalEndpoint is get-only.
        let params = NWParameters.tcp
        params.requiredInterfaceType = .loopback
        self.listener = try NWListener(using: params, on: p)
    }

    func start() {
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                FileHandle.standardError.write(
                    Data("PIPETTE_COREAI_READY port=\(self.port) model=\(self.engine.name)\n".utf8)
                )
            case .failed(let error):
                FileHandle.standardError.write(
                    Data("PIPETTE_COREAI_ERROR bind failed: \(error)\n".utf8)
                )
                exit(1)
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] conn in
            guard let self else { conn.cancel(); return }
            conn.start(queue: .global())
            self.handle(conn)
        }
        listener.start(queue: .global())
    }

    private func handle(_ conn: NWConnection) {
        let buf = ReadBuffer()
        func read() {
            conn.receive(minimumIncompleteLength: 1, maximumLength: 1_048_576) { data, _, isComplete, error in
                if let data, !data.isEmpty { buf.data.append(data) }
                if self.requestComplete(buf.data) {
                    self.dispatch(buf.data, to: conn)
                    return
                }
                if error != nil || isComplete { conn.cancel(); return }
                read()
            }
        }
        read()
    }

    private func requestComplete(_ data: Data) -> Bool {
        guard let s = String(data: data, encoding: .utf8),
              let headerEnd = s.range(of: "\r\n\r\n") else {
            return false
        }
        let header = s[s.startIndex..<headerEnd.lowerBound]
        // The fixed pipette client sends the whole body in one write, but honor
        // Content-Length anyway so a request whose body arrives in a later TCP
        // segment is still read correctly.
        let length = header
            .components(separatedBy: "\r\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { $0.lowercased().hasPrefix("content-length:") }
            .flatMap { $0.split(separator: ":").last }
            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard let length else { return true } // no body expected; headers are enough
        return data.count >= headerEnd.upperBound.utf16Offset(in: s) + length
    }

    private func dispatch(_ raw: Data, to conn: NWConnection) {
        guard let s = String(data: raw, encoding: .utf8),
              let headerEnd = s.range(of: "\r\n\r\n"),
              let requestLine = s.components(separatedBy: "\r\n").first?.components(separatedBy: " "),
              requestLine.count >= 2 else {
            respond("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n", to: conn)
            return
        }
        let method = requestLine[0]
        let path = requestLine[1]
        let body = String(s[headerEnd.upperBound...])

        if method == "GET" && path == "/health" {
            respond(json(["ok": true]), to: conn)
        } else if method == "POST" && path == "/shutdown" {
            respond(json(["ok": true]), to: conn)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { exit(0) }
        } else if method == "POST" {
            Task {
                let response = await self.post(path, body: body)
                self.respond(response, to: conn)
            }
        } else {
            respond(json(["error": "unknown endpoint: \(method) \(path)"], status: 404), to: conn)
        }
    }

    private func post(_ path: String, body: String) async -> String {
        do {
            guard let obj = try JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any] else {
                return json(["error": "request body must be a JSON object"], status: 400)
            }
            switch path {
            case "/prefill_throughput":
                guard let n = obj["prompt_tokens"] as? Int, n > 0 else {
                    return json(["error": "'prompt_tokens' must be a positive integer"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: n), maxTokens: 1)
                return json(["prompt_tps": r.promptTps, "prompt_tokens": n])
            case "/decode_throughput":
                guard let p = obj["prompt_tokens"] as? Int, p > 0,
                      let d = obj["decode_tokens"] as? Int, d > 0 else {
                    return json(["error": "'prompt_tokens'/'decode_tokens' positive ints required"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: p), maxTokens: d)
                return json(["generation_tps": r.genTps, "decode_tokens": d])
            case "/max_memory_usage":
                guard let p = obj["prompt_tokens"] as? Int, p > 0,
                      let d = obj["decode_tokens"] as? Int, d > 0 else {
                    return json(["error": "'prompt_tokens'/'decode_tokens' positive ints required"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: p), maxTokens: d)
                return json(["prompt_tokens": r.promptTokens, "completion_tokens": r.completionTokens])
            case "/end_to_end_latency":
                guard let n = obj["prompt_tokens"] as? Int, n > 0,
                      let d = obj["decode_tokens"] as? Int, d > 0 else {
                    return json(["error": "'prompt_tokens'/'decode_tokens' positive ints required"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: n), maxTokens: d)
                return json([
                    "total_ms": r.genSeconds * 1000.0,
                    "prompt_tokens": r.promptTokens,
                    "completion_tokens": r.completionTokens,
                ])
            default:
                return json(["error": "unknown endpoint: POST \(path)"], status: 404)
            }
        } catch {
            return json(["error": "\(error)"], status: 500)
        }
    }

    private func json(_ obj: [String: Any], status: Int = 200) -> String {
        let body = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
        let statusText = status == 200 ? "OK" : "Error"
        return "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n" + String(data: body, encoding: .utf8)!
    }

    private func respond(_ response: String, to conn: NWConnection) {
        guard let data = response.data(using: .utf8) else { conn.cancel(); return }
        conn.send(content: data, completion: .contentProcessed { _ in conn.cancel() })
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

@main
struct SidecarMain {
    static func main() async {
        do {
            let args = try parseArgs(CommandLine.arguments)
            let dir = URL(fileURLWithPath: args.modelDir, isDirectory: true)
            FileHandle.standardError.write(Data("PIPETTE_COREAI_LOADING \(args.modelDir)\n".utf8))
            let engine = try await CoreAIEngine(bundleDir: dir, seed: args.seed)
            let server = try SidecarServer(port: args.port, engine: engine)
            server.start()
            // Suspend forever so the process stays alive; the NWListener's global
            // queue keeps serving. Never call dispatchMain()/RunLoop.run() from an
            // async @main — dispatchMain() traps there.
            await withCheckedContinuation { (_: CheckedContinuation<Void, Never>) in
                // never resumed
            }
        } catch {
            FileHandle.standardError.write(Data("PIPETTE_COREAI_ERROR \(error)\n".utf8))
            exit(1)
        }
    }
}
