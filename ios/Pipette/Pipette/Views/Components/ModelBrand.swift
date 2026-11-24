import ImageIO
import SwiftUI
import UIKit

/// Maps a model to its vendor brand for the logo shown in selection rows.
///
/// Detection keys off the model name and repo text, not the HF org: redistributors
/// (e.g. `unsloth/gemma-...`) repackage other vendors' models.
enum ModelBrand {
    case liquid, google, ibm, qwen, meta, mistral, microsoft, deepseek, unknown

    static func detect(name: String, hfRepo: String?) -> ModelBrand {
        let hay = "\(name) \(hfRepo ?? "")".lowercased()
        if hay.contains("lfm") || hay.contains("liquid") { return .liquid }
        if hay.contains("gemma") { return .google }
        if hay.contains("granite") || hay.contains("ibm") { return .ibm }
        if hay.contains("qwen") { return .qwen }
        if hay.contains("llama") || hay.contains("meta") { return .meta }
        if hay.contains("mistral") || hay.contains("ministral") || hay.contains("mixtral") { return .mistral }
        if hay.contains("phi") { return .microsoft }
        if hay.contains("deepseek") { return .deepseek }
        return .unknown
    }

    var assetName: String? {
        switch self {
        case .unknown: return nil
        case .liquid: return "brand-liquid"
        case .google: return "brand-google"
        case .ibm: return "brand-ibm"
        case .qwen: return "brand-qwen"
        case .meta: return "brand-meta"
        case .mistral: return "brand-mistral"
        case .microsoft: return "brand-microsoft"
        case .deepseek: return "brand-deepseek"
        }
    }
}

private struct CatalogLogoOwnerKey: EnvironmentKey {
    static let defaultValue: ModelShortList.Owner? = nil
}

extension EnvironmentValues {
    var catalogLogoOwner: ModelShortList.Owner? {
        get { self[CatalogLogoOwnerKey.self] }
        set { self[CatalogLogoOwnerKey.self] = newValue }
    }
}

/// Show the catalog logo when available, or use the bundled brand logo or a neutral symbol.
/// The view requests the logo when it appears on screen.
struct BrandLogoView: View {
    @Environment(\.ws) private var ws
    @Environment(\.catalogLogoOwner) private var owner

    let brand: ModelBrand
    var logoDigest: String?
    var size: CGFloat = 28
    @State private var remote: (key: RemoteLogos.Key, image: UIImage)?
    @State private var isOnScreen = false

    var body: some View {
        let key = logoDigest.flatMap { digest in
            owner.map { RemoteLogos.Key(workspace: ws, owner: $0, digest: digest) }
        }
        Group {
            if let image = key.flatMap({ key in
                remote?.key == key ? remote?.image : RemoteLogos.shared.loaded(key)
            }) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
            } else if let asset = brand.assetName, UIImage(named: asset) != nil {
                Image(asset)
                    .resizable()
                    .scaledToFit()
            } else {
                Image(systemName: "cube")
                    .resizable()
                    .scaledToFit()
                    .padding(3)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: size, height: size)
        .onScrollVisibilityChange { isOnScreen = $0 }
        .task(id: isOnScreen ? key : nil) {
            guard isOnScreen, let key else { return }
            let image = await RemoteLogos.shared.load(key, workspace: ws)
            guard !Task.isCancelled else { return }
            remote = image.map { (key, $0) }
        }
    }
}

/// Cache logos and permanent failures for the app run. Retry transient failures on a later request.
final class RemoteLogos {
    static let shared = RemoteLogos()
    private static let retryCooldown: Duration = .seconds(30)
    private nonisolated static let maxImageBytes = 2 * 1024 * 1024
    private nonisolated static let maxImageDimension = 512

    nonisolated struct Key: Hashable, Sendable {
        let root: URL
        let owner: ModelShortList.Owner
        let digest: String

        init(workspace: PipetteWorkspace, owner: ModelShortList.Owner, digest: String) {
            root = workspace.root.standardizedFileURL
            self.owner = owner
            self.digest = digest
        }
    }

    private enum FetchResult {
        case image(UIImage)
        case unavailable
        case transientFailure
        case identityChanged
    }

    private enum Entry {
        case loading(Task<UIImage?, Never>)
        case image(UIImage)
        case unavailable
        case retryAfter(ContinuousClock.Instant)
    }

    private var entries: [Key: Entry] = [:]

    /// Return the cached logo so a row can show it immediately when the row reappears.
    func loaded(_ key: Key) -> UIImage? {
        guard case .image(let image) = entries[key] else { return nil }
        return image
    }

    func load(_ key: Key, workspace: PipetteWorkspace, http: HttpClient = .shared) async -> UIImage? {
        guard key.root == workspace.root.standardizedFileURL else { return nil }
        switch entries[key] {
        case .image(let image): return image
        case .loading(let task): return await task.value
        case .unavailable: return nil
        case .retryAfter(let deadline) where ContinuousClock.now < deadline: return nil
        case .retryAfter, nil: break
        }

        let task = Task<UIImage?, Never> {
            switch await Self.fetch(key, workspace: workspace, http: http) {
            case .image(let image):
                entries[key] = .image(image)
                return image
            case .unavailable:
                entries[key] = .unavailable
                return nil
            case .transientFailure:
                entries[key] = .retryAfter(ContinuousClock.now + Self.retryCooldown)
                return nil
            case .identityChanged:
                entries[key] = nil
                return nil
            }
        }
        entries[key] = .loading(task)
        return await task.value
    }

    private nonisolated static func isRetryableStatus(_ status: Int) -> Bool {
        status == 408 || status == 429 || (500..<600).contains(status)
    }

    private nonisolated static func fetch(_ key: Key, workspace: PipetteWorkspace, http: HttpClient) async -> FetchResult {
        let digest = key.digest
        let registration = workspace.registration
        guard let session = registration.withIdentityLock({ () -> MgmtSession? in
            guard let session = try? registration.mgmtSession(),
                  (try? ModelShortList.Owner(session)) == key.owner else { return nil }
            return session
        }) else { return .identityChanged }

        do {
            let logo = try await PipetteMgmtClient.ClientApi.familyLogo(
                server: session.serverUrl, http: http, auth: session.auth, sha256: digest)
            guard let url = URL(string: logo.logoUri), ["data", "https"].contains(url.scheme) else {
                AppLog.storage.warning("family logo URI unsupported: digest=\(digest)")
                return .unavailable
            }
            let (data, response) = try await URLSession.shared.boundedData(
                for: URLRequest(url: url), limit: maxImageBytes
            ) { response in
                (response as? HTTPURLResponse).map { $0.statusCode == 200 } ?? true
            }
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                AppLog.storage.warning(
                    "family logo download rejected: digest=\(digest) status=\(http.statusCode)")
                return isRetryableStatus(http.statusCode) ? .transientFailure : .unavailable
            }
            guard let data else {
                AppLog.storage.warning("family logo exceeded byte limit: digest=\(digest)")
                return .unavailable
            }
            guard let image = thumbnail(data) else {
                AppLog.storage.warning("family logo not decodable: digest=\(digest)")
                return .unavailable
            }
            return .image(image)
        } catch {
            AppLog.storage.warning("family logo fetch failed: digest=\(digest) error=\(error)")
            if let error = error as? PipetteMgmtClient.Error {
                let retryable = error.isTransient || error.httpStatusCode.map(isRetryableStatus) == true
                return retryable ? .transientFailure : .unavailable
            }
            if let error = error as? URLError {
                switch error.code {
                case .timedOut, .notConnectedToInternet, .networkConnectionLost, .cannotConnectToHost,
                     .cannotFindHost, .dnsLookupFailed, .resourceUnavailable:
                    return .transientFailure
                default: break
                }
            }
            return .unavailable
        }
    }

    private nonisolated static func thumbnail(_ data: Data) -> UIImage? {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions) else { return nil }
        let options = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxImageDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ] as CFDictionary
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options),
              image.width <= maxImageDimension, image.height <= maxImageDimension else { return nil }
        return UIImage(cgImage: image)
    }
}
