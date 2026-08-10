//! Upstream ggml-org/llama.cpp GitHub release listing for the CLI.

use serde::Deserialize;

use pipette_http::HttpClient;
use pipette_plan_types::LlamaCppFlavor;

const GITHUB_API_BASE: &str = "https://api.github.com/repos/ggml-org/llama.cpp";

#[derive(Debug, Deserialize)]
pub struct GitHubRelease {
    pub tag_name: String,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub assets: Vec<GitHubReleaseAsset>,
}

#[derive(Debug, Deserialize)]
pub struct GitHubReleaseAsset {
    pub name: String,
}

pub fn release_asset_available(release: &GitHubRelease, flavor: &LlamaCppFlavor) -> bool {
    flavor
        .release_asset_name(&release.tag_name)
        .map(|asset_name| release.assets.iter().any(|asset| asset.name == asset_name))
        .unwrap_or(false)
}

pub fn github_releases(http: &HttpClient, limit: usize) -> anyhow::Result<Vec<GitHubRelease>> {
    let mut releases: Vec<GitHubRelease> = http
        .json_request(
            reqwest::Method::GET,
            &format!("{GITHUB_API_BASE}/releases?per_page={limit}"),
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
    releases.truncate(limit);
    Ok(releases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_asset_available_requires_matching_asset_name() {
        let release = GitHubRelease {
            tag_name: "b9305".to_string(),
            published_at: None,
            assets: vec![GitHubReleaseAsset {
                name: "llama-b9305-bin-macos-arm64-kleidiai.tar.gz".to_string(),
            }],
        };

        assert!(release_asset_available(
            &release,
            &LlamaCppFlavor::MacosArm64Kleidiai
        ));
        assert!(!release_asset_available(
            &release,
            &LlamaCppFlavor::MacosArm64
        ));
    }
}
