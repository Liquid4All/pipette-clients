//! What a benchmark records of the server that ran it, beyond what every
//! engine does. No runtime flags: the provider is install-time.

use std::path::Path;

use crate::server;

/// The server's command as a row records it: the script by name, since its
/// path is a per-run temp copy.
pub(crate) fn command(python: &Path) -> Vec<String> {
    vec![
        python.display().to_string(),
        server::SCRIPT_FILENAME.to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_script_is_named_without_its_path() {
        assert_eq!(
            command(Path::new("/venv/bin/python")),
            ["/venv/bin/python", "pipette_onnx_genai_server.py"]
        );
    }
}
