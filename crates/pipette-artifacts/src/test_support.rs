//! Helpers shared by this crate's unit tests.

use std::io::Write;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// What the log subscriber writes, for reading an event back.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl Written {
    fn text(&self) -> anyhow::Result<String> {
        let bytes = self.0.lock().map_err(|e| anyhow::anyhow!("{e}"))?.clone();
        Ok(String::from_utf8(bytes)?)
    }
}

impl Write for Written {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` and return its result with the text of every event that it logged.
pub(crate) fn logged_text<R>(f: impl FnOnce() -> R) -> anyhow::Result<(R, String)> {
    let written = Written::default();
    let writer = written.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let result = tracing::subscriber::with_default(subscriber, f);
    Ok((result, written.text()?))
}

/// `declared` with every vault source under a `custom_label`, found by the tag
/// rule pipette-mgmt drops the label by.
pub(crate) fn labelled<T: Serialize + DeserializeOwned>(declared: &T) -> anyhow::Result<T> {
    fn label(value: &mut Value) {
        match value {
            Value::Object(fields) => {
                let vault = ["source", "type"].into_iter().any(|key| {
                    fields
                        .get(key)
                        .and_then(Value::as_str)
                        .is_some_and(|tag| tag.starts_with("vault_"))
                });
                if vault {
                    fields.insert("custom_label".to_owned(), "acme LFM3 q4".into());
                }
                fields.values_mut().for_each(label);
            }
            Value::Array(items) => items.iter_mut().for_each(label),
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
    let mut value = serde_json::to_value(declared)?;
    label(&mut value);
    Ok(serde_json::from_value(value)?)
}
