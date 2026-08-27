//! Bounded capture of subprocess stdout/stderr lines.

use std::sync::{Arc, Mutex};

/// Append `line` plus a newline, then trim whole lines off the front once the
/// buffer exceeds `cap` bytes. Cutoff is always a char boundary so a non-ASCII
/// sidecar line cannot panic on `drain`.
pub fn push_capped_line(buf: &Arc<Mutex<String>>, line: &str, cap: usize) {
    let mut buf = buf.lock().unwrap_or_else(|e| e.into_inner());
    buf.push_str(line);
    buf.push('\n');
    if buf.len() <= cap {
        return;
    }
    let over = floor_char_boundary(&buf, buf.len() - cap);
    if over == 0 || over >= buf.len() {
        return;
    }
    let cutoff = buf[over..].find('\n').map(|i| over + i + 1).unwrap_or(over);
    let cutoff = if cutoff >= buf.len() { over } else { cutoff };
    buf.drain(..cutoff);
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_from_the_front_on_char_boundaries() {
        let buf = Arc::new(Mutex::new(String::new()));
        // 2-byte UTF-8 char. A byte-index cut in the middle used to panic.
        push_capped_line(&buf, "éééé", 5);
        let s = buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert!(s.is_char_boundary(0));
        assert!(!s.is_empty());
        assert!(s.contains('é'));
    }

    #[test]
    fn keeps_the_tail() {
        let buf = Arc::new(Mutex::new(String::new()));
        push_capped_line(&buf, "aaaa", 10);
        push_capped_line(&buf, "bbbb", 10);
        push_capped_line(&buf, "cccc", 10);
        let s = buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert!(s.contains("cccc"));
        assert!(!s.contains("aaaa"));
    }
}
