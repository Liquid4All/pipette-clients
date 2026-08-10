import SwiftUI
import UIKit

/// Maps a model family to its vendor brand for the logo shown in selection rows.
///
/// Detection keys off the model name/repo text rather than the HF org, because
/// redistributors (e.g. `unsloth/gemma-...`) repackage other vendors' models —
/// the org would mislabel them.
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

    /// Asset-catalog image name for this brand, or nil when there's no logo to
    /// show (unknown vendor). The matching imageset lives in Assets.xcassets.
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

/// Leading icon for a model row. Renders the brand logo when its asset has been
/// added to the catalog; otherwise falls back to a neutral SF Symbol so the
/// row layout stays stable before real logos are dropped in.
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

/// Rounded checkbox matching the job wizard mockup: filled with the foreground
/// color and a white check when on, an outlined empty square when off.
struct WizardCheckbox: View {
    let isOn: Bool
    var isMixed: Bool = false
    var size: CGFloat = 26

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill((isOn || isMixed) ? Color.primary : Color.clear)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder((isOn || isMixed) ? Color.clear : Color(.systemGray3), lineWidth: 1.5)
            )
            .overlay(
                Image(systemName: isMixed ? "minus" : "checkmark")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Color(.systemBackground))
                    .opacity((isOn || isMixed) ? 1 : 0)
            )
            .frame(width: size, height: size)
    }
}
