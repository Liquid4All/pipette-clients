//! What a benchmark records of the server that ran it, beyond what every
//! engine does.

use std::path::Path;

use crate::server;

/// The server's command as a row records it: the script by name with no
/// arguments, since the script is a per-run temp copy, the port changes every
/// run, and the properties hold this host's `CACHE_DIR`.
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
    fn the_script_is_named_without_its_path_or_arguments() {
        assert_eq!(
            command(Path::new("/venv/bin/python")),
            ["/venv/bin/python", "pipette_openvino_server.py"]
        );
    }
}
