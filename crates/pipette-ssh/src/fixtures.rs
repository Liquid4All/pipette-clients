//! Real `ssh-keygen -t ed25519` output for two throwaway keys, one with a
//! passphrase and one without.
//!
//! Shared by the parser's tests and the inventory's, which need the same key
//! readable from a file and held by an agent at once.

/// `ssh-keygen -t ed25519 -N '' -C plain@fixture`.
pub(crate) const PLAIN: &str = "\
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDSPmqwrdKAo++4+x5I5kfqPqrl42H1g7LyaDNmyQ4VHAAAAJDTqPDe06jw
3gAAAAtzc2gtZWQyNTUxOQAAACDSPmqwrdKAo++4+x5I5kfqPqrl42H1g7LyaDNmyQ4VHA
AAAED8W19NfTim3iQWwb8wTU1JpYRYDTYWr8vJLvdDJGLnVdI+arCt0oCj77j7HkjmR+o+
quXjYfWDsvJoM2bJDhUcAAAADXBsYWluQGZpeHR1cmU=
-----END OPENSSH PRIVATE KEY-----
";

/// Its `.pub` line, and the fingerprint `ssh-keygen -l` prints for it.
pub(crate) const PLAIN_PUB: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINI+arCt0oCj77j7HkjmR+o+quXjYfWDsvJoM2bJDhUc plain@fixture";
pub(crate) const PLAIN_FINGERPRINT: &str = "SHA256:eJQWmJmce1mUjjLjicIm/xIjF6YmW2R9Aoz0r6yyFjo";

/// The same command with `-N passphrase -C locked@fixture`: the shape most
/// people's `~/.ssh/id_ed25519` actually has.
pub(crate) const LOCKED: &str = "\
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAACmFlczI1Ni1jdHIAAAAGYmNyeXB0AAAAGAAAABA6jl/hcI
JejfCeiCpw5nnDAAAAGAAAAAEAAAAzAAAAC3NzaC1lZDI1NTE5AAAAILWDdeytL0Y7BuUu
OiyByscQWR59pZk63ZZRJ5JlnNnBAAAAoINh52cRVnrNeu2jOc87tIgw6pA3r8svHLoTRh
D19jFKKLPERln5Utj85/DYbm5rwn0dt/KKeDkM6lIaNGPA15WfnOUHkOfkEFFfGpBXgLN2
BZtVID4zVRXDyx9z1XbKCW0A0xwgQOG6kSWo8HjzEM0OzdHUeFnyqHDEe/Cgd1vwB0cQRq
Ys6dFVzd4u3LCzCUp9SReN23yLhrcPciHDQoM=
-----END OPENSSH PRIVATE KEY-----
";

pub(crate) const LOCKED_PUB: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILWDdeytL0Y7BuUuOiyByscQWR59pZk63ZZRJ5JlnNnB locked@fixture";
pub(crate) const LOCKED_FINGERPRINT: &str = "SHA256:QUloKa9s3Y3XV8WipN8nHOkP7Id/1bax/2AwR55OifM";
