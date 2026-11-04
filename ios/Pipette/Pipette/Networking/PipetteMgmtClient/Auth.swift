import CryptoKit
import Foundation

/// Signs a request to the management server as a device (the `v1` payload). The
/// app never authenticates as an operator, so the crate's `UserIdentity` is not
/// mirrored.
///
/// The client id (in `registration.json`) and the key (in the Keychain) are useless
/// apart, so they travel as one value. Not `Codable`, as ``PrivateKey`` is not, so
/// it can never be persisted or logged.
nonisolated struct ClientIdentity: Hashable, Sendable, SignsRequest {
    /// Typed to the wire: a padded or empty id signs as a client the server
    /// cannot resolve.
    let clientId: ClientId
    let privateKeyHex: PrivateKey
}

nonisolated extension ClientIdentity {
    nonisolated func signedHeaders(
        method: String,
        pathAndQuery: String
    ) throws(PipetteMgmtClient.Error) -> [(name: String, value: String)] {
        let stamp = PipetteMgmtClient.Auth.RequestStamp()
        let payload = PipetteMgmtClient.Auth.clientPayload(
            method: method,
            pathAndQuery: pathAndQuery,
            timestamp: stamp.timestamp,
            clientId: clientId.value,
            nonce: stamp.nonce
        )
        return try stamp.seal(
            headers: [(name: "X-Client-Id", value: clientId.value)],
            payload: payload,
            privateKey: privateKeyHex
        )
    }
}

/// How the ``PipetteMgmtClient/Call`` pipeline signs. No endpoint takes it: it
/// exists because the signature covers the target `Call` derives from the base URL,
/// so signing happens inside `Call`.
nonisolated protocol SignsRequest {
    /// All ASCII, so the transport can set them verbatim.
    ///
    /// `pathAndQuery` must be the target the server receives, with any base-URL
    /// path prefix and the query: the server verifies against its own
    /// `uri.path_and_query()`.
    func signedHeaders(
        method: String,
        pathAndQuery: String
    ) throws(PipetteMgmtClient.Error) -> [(name: String, value: String)]
}

/// The signing scheme: the crate's `auth.rs` free functions. Not mirrored:
/// `generate_keypair` (`KeychainHelper.generatePendingSigningKeyPair` has the
/// Keychain mint the key), and the user-only `key_id_for`.
nonisolated extension PipetteMgmtClient {
    enum Auth {
        /// Held together so the timestamp and nonce that were signed are the
        /// ones ``seal(headers:payload:privateKey:)`` puts on the wire.
        struct RequestStamp {
            let timestamp: String
            let nonce: String

            init() {
                timestamp = Auth.rfc3339Timestamp()
                nonce = Auth.generateNonce()
            }

            func seal(
                headers: [(name: String, value: String)],
                payload: String,
                privateKey: PrivateKey
            ) throws(PipetteMgmtClient.Error) -> [(name: String, value: String)] {
                let signature = try Auth.sign(privateKey: privateKey, payload: payload)
                return headers + [
                    (name: "X-Timestamp", value: timestamp),
                    (name: "X-Nonce", value: nonce),
                    (name: "X-Signature", value: signature),
                ]
            }
        }

        /// A `Sendable` format style, not an `ISO8601DateFormatter`: that is a
        /// mutable class, unsafe to share between signing threads.
        private static let rfc3339 = Date.ISO8601FormatStyle(
            includingFractionalSeconds: true, timeZone: .gmt)

        /// With fractional seconds, as the crate stamps them (milliseconds here,
        /// nanoseconds there; both are RFC 3339, which the server parses).
        static func rfc3339Timestamp() -> String {
            Date().formatted(rfc3339)
        }

        /// 16 CSPRNG bytes (`SystemRandomNumberGenerator`), lowercase hex. Hex
        /// because the payload is newline-delimited: a nonce with a newline could
        /// forge a field boundary. 128 bits keeps fleet-wide collisions negligible
        /// for the server's replay cache.
        static func generateNonce() -> String {
            Data((0..<16).map { _ in UInt8.random(in: .min ... .max) }).hexEncodedString()
        }

        /// The client `v1` payload (the mgmt repo's `docs/authentication.md` §2.1).
        /// The nonce makes a signature single-use; the body is not covered.
        static func clientPayload(
            method: String,
            pathAndQuery: String,
            timestamp: String,
            clientId: String,
            nonce: String
        ) -> String {
            "v1\n\(method)\n\(pathAndQuery)\n\(timestamp)\n\(clientId)\n\(nonce)"
        }

        /// ``PrivateKey``'s initializer already refuses both failures; they are
        /// mapped here rather than trapped on, as a newtype holds the string, not
        /// the seed.
        static func signingKey(
            _ privateKey: PrivateKey
        ) throws(PipetteMgmtClient.Error) -> Curve25519.Signing.PrivateKey {
            guard let bytes = Data(hexEncoded: privateKey.value) else {
                throw .invalidPrivateKey(.notHex)
            }
            return try signingKey(seed: bytes)
        }

        /// Length checked first, so a wrong-size seed is named as that rather than
        /// as CryptoKit's "the curve refused this key". Takes bytes so a test can
        /// reach the refusal ``PrivateKey`` already prevents.
        static func signingKey(
            seed bytes: Data
        ) throws(PipetteMgmtClient.Error) -> Curve25519.Signing.PrivateKey {
            guard bytes.count == privateKeyLengthBytes else {
                throw .invalidPrivateKey(.length(found: bytes.count))
            }
            guard let key = try? Curve25519.Signing.PrivateKey(rawRepresentation: bytes) else {
                throw .invalidPrivateKey(.notAKey)
            }
            return key
        }

        /// The public half is never stored, so this recovers the value the server
        /// has on record. Named as in `auth.rs`, so a reader finds it there.
        static func publicKeyHexFromPrivate(
            _ privateKey: PrivateKey
        ) throws(PipetteMgmtClient.Error) -> PublicKey {
            let key = try signingKey(privateKey)
            do {
                return try PublicKey(key.publicKey.rawRepresentation.hexEncodedString())
            } catch {
                // The crate's `#[from]`, spelled out.
                throw .invalidPublicKey(error)
            }
        }

        static func sign(
            privateKey: PrivateKey,
            payload: String
        ) throws(PipetteMgmtClient.Error) -> String {
            let key = try signingKey(privateKey)
            // CryptoKit's signing throws where `ed25519_dalek`'s cannot, and only
            // for a key the curve will not use.
            guard let signature = try? key.signature(for: Data(payload.utf8)) else {
                throw .invalidPrivateKey(.notAKey)
            }
            return signature.hexEncodedString()
        }
    }
}
