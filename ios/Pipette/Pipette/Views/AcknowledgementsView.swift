import LicenseList
import SwiftUI

/// Open source attributions, reachable from Settings. Swift package licenses
/// (every SwiftPM dependency, incl. the MLX stack) are generated at build time
/// by the LicenseList plugin; non-SwiftPM bundled components (the vendored
/// llama.cpp) come from ThirdPartyLicenses.json.
struct AcknowledgementsView: View {
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight

    var body: some View {
        List {
            Section("Swift Packages") {
                NavigationLink("Swift package licenses") {
                    LicenseListView()
                        .licenseViewStyle(.withRepositoryAnchorLink)
                        .contentMargins(.bottom, pillTabBarReservedHeight, for: .scrollContent)
                        .navigationTitle("Swift Packages")
                        .navigationBarTitleDisplayMode(.inline)
                }
            }

            Section("Bundled Components") {
                ForEach(Acknowledgements.all) { item in
                    NavigationLink {
                        AcknowledgementDetailView(acknowledgement: item)
                    } label: {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(item.name)
                                .font(.system(size: 16.5, weight: .regular))
                                .foregroundStyle(.primary)
                            Text(item.license)
                                .font(.system(size: 13, weight: .regular))
                                .foregroundStyle(Color(.systemGray))
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .contentMargins(.bottom, pillTabBarReservedHeight, for: .scrollContent)
        .navigationTitle("Open Source Licenses")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.visible, for: .navigationBar)
    }
}

private struct AcknowledgementDetailView: View {
    let acknowledgement: Acknowledgement
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight

    var body: some View {
        ScrollView {
            Text(acknowledgement.text)
                .font(.system(size: 12, weight: .regular, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
                .padding(.bottom, pillTabBarReservedHeight)
        }
        .navigationTitle(acknowledgement.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.visible, for: .navigationBar)
    }
}

#if DEBUG
#Preview("Acknowledgements") {
    NavigationStack {
        AcknowledgementsView()
    }
}
#endif
