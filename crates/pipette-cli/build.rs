use std::env;

fn main() {
    let build_version = env::var("PIPETTE_CLI_BUILD_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "dev".to_string());
    println!("cargo:rustc-env=PIPETTE_CLI_BUILD_VERSION={build_version}");
    println!("cargo:rerun-if-env-changed=PIPETTE_CLI_BUILD_VERSION");
}
