fn main() -> anyhow::Result<()> {
    // So `pipette … | head` exits cleanly instead of failing on a broken pipe.
    pipette_subprocess::reset_sigpipe();
    // adbd leaks its mDNS socket into this worker; see `close_inherited_fds`.
    #[cfg(target_os = "android")]
    {
        // SAFETY: no thread has started and nothing owns a descriptor yet.
        if let Err(err) = unsafe { pipette_subprocess::close_inherited_fds() } {
            eprintln!("warning: could not close inherited descriptors: {err}");
        }
    }
    pipette_cli::commands::run()
}
