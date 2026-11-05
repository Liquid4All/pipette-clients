use std::env;

fn main() {
    // Same stamp and CI variable as pipette-cli's build.rs, so a driver and a
    // client from one release report the same build (see `PLAN_VERSION`).
    let build_version = env::var("PIPETTE_CLI_BUILD_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "dev".to_string());
    println!("cargo:rustc-env=PIPETTE_CLI_BUILD_VERSION={build_version}");
    println!("cargo:rerun-if-env-changed=PIPETTE_CLI_BUILD_VERSION");
}
