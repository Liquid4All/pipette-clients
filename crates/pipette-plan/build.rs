use std::env;

fn main() {
    // Same stamp the client bakes in, from the same CI job-level variable, so a
    // driver and a client from one release report the same build. Local builds
    // report `dev`, which is what distinguishes a developer's run from a
    // released one — the driver's version decides which plan features parse.
    let build_version = env::var("PIPETTE_CLI_BUILD_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "dev".to_string());
    println!("cargo:rustc-env=PIPETTE_CLI_BUILD_VERSION={build_version}");
    println!("cargo:rerun-if-env-changed=PIPETTE_CLI_BUILD_VERSION");
}
