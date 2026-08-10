//! Generate an Ed25519 SSH keypair in OpenSSH formats — the private
//! `-----BEGIN OPENSSH PRIVATE KEY-----` PEM and the `ssh-ed25519 <base64>
//! <comment>` public line — without shelling out to `ssh-keygen`.
//!
//! A signing key is its 32-byte seed, so we read the seed from the OS CSPRNG
//! (`getrandom`), derive the public half with `ed25519-dalek`, and hand-serialize
//! the OpenSSH `openssh-key-v1` container (unencrypted: cipher/kdf `none`). The
//! layout is a fixed, non-crypto byte format, and [`encode`] is a pure seam over
//! (seed, checkint, comment) so a known-answer test can pin it byte-for-byte
//! against real `ssh-keygen` output.

use anyhow::Context;
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::SigningKey;

/// An Ed25519 keypair rendered in OpenSSH text formats.
pub struct SshEd25519Keypair {
    /// The `-----BEGIN OPENSSH PRIVATE KEY-----` PEM (unencrypted), trailing `\n`.
    pub private_openssh: String,
    /// The single `ssh-ed25519 <base64> <comment>` public line (no trailing `\n`).
    pub public_openssh: String,
}

/// Generate a fresh Ed25519 SSH keypair, tagged with `comment`.
pub fn generate(comment: &str) -> anyhow::Result<SshEd25519Keypair> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).context("failed to read OS entropy for the SSH seed")?;
    // `checkint` is a random guard OpenSSH writes twice and checks on decrypt; the
    // value is arbitrary, it only has to match itself.
    let mut checkint = [0u8; 4];
    getrandom::fill(&mut checkint).context("failed to read OS entropy for the checkint")?;
    encode(&seed, u32::from_be_bytes(checkint), comment)
}

/// Push a `string` in the SSH wire format: a big-endian `u32` length followed by
/// the bytes. Errors if the length doesn't fit `u32` (a comment over 4 GiB).
fn push_ssh_string(out: &mut Vec<u8>, bytes: &[u8]) -> anyhow::Result<()> {
    let len = u32::try_from(bytes.len()).context("SSH string length exceeds the u32 wire limit")?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// The `ssh-ed25519` public-key blob: `string "ssh-ed25519"` + `string <pubkey>`.
fn public_blob(pubkey: &[u8; 32]) -> anyhow::Result<Vec<u8>> {
    let mut blob = Vec::new();
    push_ssh_string(&mut blob, b"ssh-ed25519")?;
    push_ssh_string(&mut blob, pubkey)?;
    Ok(blob)
}

/// Serialize the OpenSSH private PEM + public line for `seed`, deterministically.
/// Split from [`generate`] so the encoding can be pinned against `ssh-keygen`.
fn encode(seed: &[u8; 32], checkint: u32, comment: &str) -> anyhow::Result<SshEd25519Keypair> {
    let pubkey = SigningKey::from_bytes(seed).verifying_key().to_bytes();
    let pub_blob = public_blob(&pubkey)?;

    let public_openssh = format!("ssh-ed25519 {} {}", STANDARD.encode(&pub_blob), comment);

    // The private section: two check-ints, the key type, the public key, the
    // 64-byte private key (seed ++ pubkey), the comment, then `1,2,3,…` padding
    // to the `none`-cipher block size of 8.
    let mut private_section = Vec::new();
    private_section.extend_from_slice(&checkint.to_be_bytes());
    private_section.extend_from_slice(&checkint.to_be_bytes());
    push_ssh_string(&mut private_section, b"ssh-ed25519")?;
    push_ssh_string(&mut private_section, &pubkey)?;
    let mut secret = Vec::with_capacity(64);
    secret.extend_from_slice(seed);
    secret.extend_from_slice(&pubkey);
    push_ssh_string(&mut private_section, &secret)?;
    push_ssh_string(&mut private_section, comment.as_bytes())?;
    let mut pad: u8 = 1;
    while private_section.len() % 8 != 0 {
        private_section.push(pad);
        pad += 1;
    }

    let mut container = Vec::new();
    container.extend_from_slice(b"openssh-key-v1\0");
    push_ssh_string(&mut container, b"none")?; // cipher
    push_ssh_string(&mut container, b"none")?; // kdf
    push_ssh_string(&mut container, b"")?; // kdf options
    container.extend_from_slice(&1u32.to_be_bytes()); // number of keys
    push_ssh_string(&mut container, &pub_blob)?;
    push_ssh_string(&mut container, &private_section)?;

    let body = STANDARD.encode(&container);
    let mut private_openssh = String::from("-----BEGIN OPENSSH PRIVATE KEY-----\n");
    // OpenSSH wraps the base64 body at 70 columns; the body is ASCII, so slicing
    // on byte boundaries is char-safe.
    let mut start = 0;
    while start < body.len() {
        let end = (start + 70).min(body.len());
        private_openssh.push_str(&body[start..end]);
        private_openssh.push('\n');
        start = end;
    }
    private_openssh.push_str("-----END OPENSSH PRIVATE KEY-----\n");

    Ok(SshEd25519Keypair {
        private_openssh,
        public_openssh,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    // Throwaway Ed25519 keys generated once with macOS `ssh-keygen -t ed25519`
    // (no passphrase). They have no security value — they exist only to pin our
    // encoder byte-for-byte against real OpenSSH output without running
    // `ssh-keygen` at test time. The two comment lengths land the private section
    // on different padding (`[1,2]` vs `[1,2,3,4]`).
    const FIXTURE_A_SEED: [u8; 32] = [
        0x3d, 0x04, 0xea, 0xc2, 0x9c, 0x3a, 0x7e, 0xc2, 0x93, 0xcd, 0xdf, 0xe5, 0x93, 0xf7, 0xef,
        0x7b, 0xdf, 0x21, 0x8d, 0x80, 0x9b, 0xd8, 0xb8, 0x49, 0x07, 0x86, 0xb4, 0x6f, 0x14, 0xcf,
        0xca, 0x96,
    ];
    const FIXTURE_A_PUBLIC: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMi++ASVhDVzheB9tNzoK8qOdHy1kiu9qFz7+xQjClWz pipette-ssh-fixture";
    const FIXTURE_A_PRIVATE: &str = "-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDIvvgElYQ1c4XgfbTc6CvKjnR8tZIrvahc+/sUIwpVswAAAJiLECvTixAr
0wAAAAtzc2gtZWQyNTUxOQAAACDIvvgElYQ1c4XgfbTc6CvKjnR8tZIrvahc+/sUIwpVsw
AAAEA9BOrCnDp+wpPN3+WT9+973yGNgJvYuEkHhrRvFM/Klsi++ASVhDVzheB9tNzoK8qO
dHy1kiu9qFz7+xQjClWzAAAAE3BpcGV0dGUtc3NoLWZpeHR1cmUBAg==
-----END OPENSSH PRIVATE KEY-----
";

    const FIXTURE_B_SEED: [u8; 32] = [
        0xbc, 0xd6, 0x18, 0xe5, 0x5b, 0x37, 0x23, 0x60, 0xa1, 0xb9, 0xd6, 0xae, 0x90, 0x2a, 0xf3,
        0xbe, 0x56, 0xab, 0x59, 0xe4, 0x41, 0xb2, 0x31, 0x33, 0xb7, 0x53, 0x41, 0x81, 0xfa, 0xd8,
        0x82, 0xeb,
    ];
    const FIXTURE_B_PUBLIC: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICFnSJ+yL4yMnZOrtgI8AcFWrHOy8qgWG8RKwi+PkF14 k";
    const FIXTURE_B_PRIVATE: &str = "-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACAhZ0ifsi+MjJ2Tq7YCPAHBVqxzsvKoFhvESsIvj5BdeAAAAIjmoI1x5qCN
cQAAAAtzc2gtZWQyNTUxOQAAACAhZ0ifsi+MjJ2Tq7YCPAHBVqxzsvKoFhvESsIvj5BdeA
AAAEC81hjlWzcjYKG51q6QKvO+VqtZ5EGyMTO3U0GB+tiC6yFnSJ+yL4yMnZOrtgI8AcFW
rHOy8qgWG8RKwi+PkF14AAAAAWsBAgME
-----END OPENSSH PRIVATE KEY-----
";

    /// Take the next `n` bytes off `cur`, advancing it.
    fn take<'a>(cur: &mut &'a [u8], n: usize) -> anyhow::Result<&'a [u8]> {
        let (head, rest) = cur.split_at_checked(n).context("truncated field")?;
        *cur = rest;
        Ok(head)
    }

    /// Take one length-prefixed SSH `string` off `cur`.
    fn take_string<'a>(cur: &mut &'a [u8]) -> anyhow::Result<&'a [u8]> {
        let len = u32::from_be_bytes(take(cur, 4)?.try_into()?) as usize;
        take(cur, len)
    }

    /// Decode an OpenSSH private PEM back to its `(seed, public_key)` — the test
    /// counterpart to [`encode`], so a generated key can be checked for real
    /// seed → public-key correspondence.
    fn parse_private(pem: &str) -> anyhow::Result<([u8; 32], [u8; 32])> {
        let body: String = pem
            .lines()
            .filter(|line| !line.contains("OPENSSH PRIVATE KEY"))
            .collect();
        let container = STANDARD.decode(body)?;
        let mut cur = container.as_slice();
        take(&mut cur, b"openssh-key-v1\0".len())?; // magic
        take_string(&mut cur)?; // cipher
        take_string(&mut cur)?; // kdf
        take_string(&mut cur)?; // kdf options
        take(&mut cur, 4)?; // number of keys
        take_string(&mut cur)?; // public blob
        let mut section = take_string(&mut cur)?;
        take(&mut section, 8)?; // two check-ints
        take_string(&mut section)?; // key type
        let pubkey: [u8; 32] = take_string(&mut section)?.try_into()?;
        let seed: [u8; 32] = take_string(&mut section)?[..32].try_into()?; // seed ++ pubkey
        Ok((seed, pubkey))
    }

    /// Known-answer: our encoder reproduces macOS `ssh-keygen` output byte-for-byte
    /// for each fixture (seed + checkint). This is the acceptance test — it pins
    /// the OpenSSH wire format against real keys without running `ssh-keygen`.
    #[rstest]
    #[case::padding_1_2(&FIXTURE_A_SEED, 0x8b10_2bd3, "pipette-ssh-fixture", FIXTURE_A_PUBLIC, FIXTURE_A_PRIVATE)]
    #[case::padding_1_2_3_4(&FIXTURE_B_SEED, 0xe6a0_8d71, "k", FIXTURE_B_PUBLIC, FIXTURE_B_PRIVATE)]
    fn encodes_byte_identical_to_ssh_keygen(
        #[case] seed: &[u8; 32],
        #[case] checkint: u32,
        #[case] comment: &str,
        #[case] expected_public: &str,
        #[case] expected_private: &str,
    ) -> anyhow::Result<()> {
        let keypair = encode(seed, checkint, comment)?;
        assert_eq!(keypair.public_openssh, expected_public);
        assert_eq!(keypair.private_openssh, expected_private);
        Ok(())
    }

    /// A freshly generated key is well-formed and a genuine keypair: the private
    /// seed derives the public key embedded in the private blob *and* the one in
    /// the public line, and two generations differ.
    #[test]
    fn generated_key_is_a_consistent_random_keypair() -> anyhow::Result<()> {
        let keypair = generate("dev@pipette")?;

        assert!(keypair.public_openssh.starts_with("ssh-ed25519 "));
        assert!(keypair.public_openssh.ends_with(" dev@pipette"));

        // The public key as advertised on the public line.
        let b64 = keypair
            .public_openssh
            .split(' ')
            .nth(1)
            .context("public line has a base64 field")?;
        let pub_blob = STANDARD.decode(b64)?;
        let line_pubkey: [u8; 32] = pub_blob[pub_blob.len() - 32..].try_into()?;

        // The seed + public key stored in the private blob.
        let (seed, embedded_pubkey) = parse_private(&keypair.private_openssh)?;

        // The seed derives that key, and both copies agree — a real keypair.
        let derived = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
        assert_eq!(derived, line_pubkey);
        assert_eq!(embedded_pubkey, line_pubkey);

        assert_ne!(
            generate("x")?.public_openssh,
            generate("x")?.public_openssh,
            "each generation must be random"
        );
        Ok(())
    }
}
