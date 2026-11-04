import Foundation

/// A non-SPM open source component bundled in the app (e.g. vendored llama.cpp), from
/// ThirdPartyLicenses.json; LicenseList covers Swift packages. Regenerate the JSON with
/// ios/generate-third-party-licenses.py when the llama.cpp submodule is bumped.
struct Acknowledgement: Identifiable, Decodable {
    let name: String
    let license: String
    let text: String
    var id: String { name }
}

enum Acknowledgements {
    static let all: [Acknowledgement] = {
        guard let url = Bundle.main.url(forResource: "ThirdPartyLicenses", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let entries = try? JSONDecoder().decode([Acknowledgement].self, from: data)
        else {
            assertionFailure("ThirdPartyLicenses.json missing or malformed")
            return []
        }
        return entries
    }()
}
