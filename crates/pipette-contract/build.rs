fn main() {
    // rstest lists the fixtures when the contract test compiles, so adding,
    // removing or editing one rebuilds this crate, and only this crate.
    println!("cargo:rerun-if-changed=fixtures");
}
