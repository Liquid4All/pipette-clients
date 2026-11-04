//! Where a command writes. `out` carries what the command produces (a listing,
//! a result record, a token); `err` what is said to the person watching. Kept
//! apart so a redirect captures the product alone. A trait so a command runs
//! on the process streams from `dispatch` and on buffers under test. A
//! `write_*` helper takes the writer first, since writing is its whole job;
//! anything else takes the console last, after what it operates on.

use std::io::Write;

pub trait Console {
    /// What the command produces.
    fn out(&mut self) -> &mut dyn Write;

    /// What is said about producing it.
    fn err(&mut self) -> &mut dyn Write;
}

pub struct Stdio {
    out: std::io::Stdout,
    err: std::io::Stderr,
}

impl Default for Stdio {
    fn default() -> Self {
        Self {
            out: std::io::stdout(),
            err: std::io::stderr(),
        }
    }
}

impl Console for Stdio {
    fn out(&mut self) -> &mut dyn Write {
        &mut self.out
    }

    fn err(&mut self) -> &mut dyn Write {
        &mut self.err
    }
}
