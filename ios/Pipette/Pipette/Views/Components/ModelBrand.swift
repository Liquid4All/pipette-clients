import SwiftUI
import UIKit

/// Maps a model family to its vendor brand for the logo shown in selection rows.
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

/// Leading icon for a model row: the brand logo, or a neutral SF Symbol when the
/// brand has no asset.
struct BrandLogoView: View {
    let brand: ModelBrand
    var size: CGFloat = 28

    var body: some View {
        Group {
            if let asset = brand.assetName, UIImage(named: asset) != nil {
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
    }
}
