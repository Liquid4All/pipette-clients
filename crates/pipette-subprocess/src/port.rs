//! A loopback port for a server this process is about to start.
//!
//! The server binds the port itself, after this check has released it, so the
//! port is picked where nothing else lands in that gap: outside the ranges an
//! OS hands out for `bind(0)` and for the local end of an outgoing connection.

use std::hash::{BuildHasher, RandomState};
use std::net::TcpListener;
use std::ops::RangeInclusive;

/// Below the default ephemeral ranges: Linux starts at 32768, macOS and Windows
/// at 49152.
const PORTS: RangeInclusive<u16> = 20000..=32767;
const LEN: u16 = *PORTS.end() - *PORTS.start() + 1;
/// Ports that fail to bind are skipped: held by a server, or in a block Windows
/// reserves for Hyper-V.
const TRIES: u16 = 64;
/// Wider than the 100-port blocks Hyper-V reserves, so each try lands in a
/// different one, and coprime to [`LEN`], so no port repeats.
const STRIDE: u16 = 101;

/// Every port [`pick_server_port`] tried failed to bind.
#[derive(Debug, thiserror::Error)]
#[error("no free loopback port in {PORTS:?} after {TRIES} tries")]
pub struct NoFreePortError;

/// A free loopback port for a server to bind, from a random point in [`PORTS`]
/// so parallel launchers rarely try the same one.
pub fn pick_server_port() -> Result<u16, NoFreePortError> {
    // A fresh `RandomState` is OS-seeded and differs on every call. Below
    // `LEN`, so the conversion cannot fail.
    let random = RandomState::new().hash_one(()) % u64::from(LEN);
    pick_from(u16::try_from(random).unwrap_or_default())
}

fn pick_from(offset: u16) -> Result<u16, NoFreePortError> {
    candidates(offset)
        .find(|&port| TcpListener::bind(("127.0.0.1", port)).is_ok())
        .ok_or(NoFreePortError)
}

fn candidates(offset: u16) -> impl Iterator<Item = u16> {
    let offset = offset % LEN;
    (0..TRIES).map(move |i| PORTS.start() + (offset + i * STRIDE) % LEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picked_port_is_in_the_range_and_free() -> anyhow::Result<()> {
        let port = pick_server_port()?;
        assert!(PORTS.contains(&port), "{port}");
        TcpListener::bind(("127.0.0.1", port))?;
        Ok(())
    }

    #[test]
    fn a_held_port_is_skipped() -> anyhow::Result<()> {
        let held_port = pick_server_port()?;
        let _held = TcpListener::bind(("127.0.0.1", held_port))?;
        let port = pick_from(held_port - PORTS.start())?;
        assert_ne!(port, held_port);
        assert!(PORTS.contains(&port), "{port}");
        Ok(())
    }

    #[test]
    fn candidates_wrap_to_the_start_of_the_range() {
        let ports: Vec<u16> = candidates(LEN - 1).take(2).collect();
        assert_eq!(ports, [32767, 20100]);
    }

    #[test]
    fn candidates_are_distinct() {
        let ports: std::collections::HashSet<u16> = candidates(0).collect();
        assert_eq!(ports.len(), usize::from(TRIES));
    }
}
