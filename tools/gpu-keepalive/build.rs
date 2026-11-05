//! Bakes the version into the binary, and on Windows into its file properties.
//!
//! `GPU_KEEPALIVE_VERSION` is MAJOR.MINOR.PATCH and doubles as the MSI
//! ProductVersion, so each field has to fit 255.255.65535. CI maps the release
//! version (2026.09.11-<sha>) to 26.9.11 and passes the full string as
//! `GPU_KEEPALIVE_BUILD`, which `gpu-keepalive --version` reports.

fn main() {
    println!("cargo:rerun-if-env-changed=GPU_KEEPALIVE_VERSION");
    println!("cargo:rerun-if-env-changed=GPU_KEEPALIVE_BUILD");
    let version = std::env::var("GPU_KEEPALIVE_VERSION").unwrap_or_else(|_| "0.0.0".to_owned());
    let build = std::env::var("GPU_KEEPALIVE_BUILD").unwrap_or_else(|_| "dev".to_owned());
    let fields: Vec<u64> = version
        .split('.')
        .map(|field| field.parse().unwrap_or(u64::MAX))
        .collect();
    let [major, minor, patch] = fields[..] else {
        panic!("GPU_KEEPALIVE_VERSION must be MAJOR.MINOR.PATCH, got '{version}'");
    };
    assert!(
        major <= 255 && minor <= 255 && patch <= 65535,
        "GPU_KEEPALIVE_VERSION fields must fit 255.255.65535, got '{version}'"
    );
    println!("cargo:rustc-env=GPU_KEEPALIVE_VERSION={version}");
    println!("cargo:rustc-env=GPU_KEEPALIVE_BUILD={build}");

    // Host-gated rather than target-gated: the resource compiler is a Windows
    // SDK tool, so a cross-target `cargo check` from macOS still type-checks.
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource
            .set("CompanyName", "Liquid AI, Inc.")
            .set("FileDescription", "Pipette GPU Keepalive")
            .set("ProductName", "Pipette GPU Keepalive")
            .set("InternalName", "gpu-keepalive")
            .set("OriginalFilename", "gpu-keepalive.exe")
            .set("FileVersion", &build)
            .set("ProductVersion", &version)
            .set_version_info(
                winresource::VersionInfo::FILEVERSION,
                (major << 48) | (minor << 32) | (patch << 16),
            )
            .set_version_info(
                winresource::VersionInfo::PRODUCTVERSION,
                (major << 48) | (minor << 32) | (patch << 16),
            );
        resource.compile().expect("compiling the version resource");
    }
}
