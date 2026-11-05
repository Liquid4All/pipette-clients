//! Every `auth` verb is reachable through the parser the binary runs. Asserted
//! through [`Cli`] because unit tests call the verb functions directly and stay
//! green when `Command` has no variant to reach them.
//!
//! In `tests/` because these flags read the environment, and one test binary
//! means nothing races the `remove_var` calls.

use clap::Parser;

use pipette_ctl::auth::{AuthCommand, KeysCommand};
use pipette_ctl::cli::{Cli, Command};

fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(["pipette-ctl"].iter().chain(args))
}

/// One row of the verb table: the argv to parse, and the verb it has to
/// produce.
type VerbCase<'a> = (&'a [&'a str], fn(&AuthCommand) -> bool);

#[test]
fn every_auth_verb_routes_through_the_cli() -> anyhow::Result<()> {
    [
        "PIPETTE_MGMT_SERVER",
        "PIPETTE_USER_PREAUTH_KEY",
        "PIPETTE_WORK_DIR",
    ]
    .into_iter()
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    .for_each(|key| unsafe { std::env::remove_var(key) });

    // A syntactically valid Ed25519 public key: `keys add` parses the hex and
    // the length at the flag, so a placeholder would fail for the wrong reason.
    let public_key = "a3f8".repeat(16);
    // Each row names the verb it must parse to, not just that it reached
    // `auth`: a mis-wiring that routed every subcommand to one verb would
    // satisfy the weaker assertion.
    let verbs: [VerbCase; 8] = [
        (
            &[
                "auth",
                "register",
                "--user-preauth-key",
                "user_preauth_k1.secret",
                "--key-details",
                "work laptop",
            ],
            |verb| matches!(verb, AuthCommand::Register(_)),
        ),
        (
            &["auth", "me"],
            |verb| matches!(verb, AuthCommand::Me(args) if !args.local),
        ),
        (
            &["auth", "me", "--local"],
            |verb| matches!(verb, AuthCommand::Me(args) if args.local),
        ),
        (&["auth", "adopt", "--user-id", "user_0192f0c1"], |verb| {
            matches!(verb, AuthCommand::Adopt(_))
        }),
        // The mgmt app's sign-in page prints this argv for the reader to run,
        // so renaming the verb or any of its flags breaks the instructions there.
        (
            &[
                "auth",
                "keys",
                "add",
                "--public-key",
                &public_key,
                "--key-details",
                "console app, Chrome on macOS",
                "--expires-in",
                "7d",
            ],
            |verb| matches!(verb, AuthCommand::Keys(args) if matches!(&args.command, KeysCommand::Add(add) if add.expires_in.is_some())),
        ),
        (
            &["auth", "keys", "revoke", "userkey_9f2c"],
            |verb| matches!(verb, AuthCommand::Keys(args) if matches!(args.command, KeysCommand::Revoke(_))),
        ),
        (
            &["auth", "keys", "available"],
            |verb| matches!(verb, AuthCommand::Keys(args) if matches!(args.command, KeysCommand::Available)),
        ),
        (&["auth", "pubkey"], |verb| {
            matches!(verb, AuthCommand::Pubkey)
        }),
    ];

    verbs.into_iter().try_for_each(|(argv, expected)| {
        let cli = parse(argv).map_err(|error| anyhow::anyhow!("{argv:?} should parse: {error}"))?;
        let Command::Auth(args) = cli.command else {
            anyhow::bail!("{argv:?} should route to the auth verbs");
        };
        anyhow::ensure!(
            expected(&args.command),
            "{argv:?} parsed to the wrong verb: {:?}",
            args.command
        );
        anyhow::Ok(())
    })?;

    let cli = parse(&["--work-dir", "/tmp/plan", "auth", "pubkey"])
        .map_err(|error| anyhow::anyhow!("--work-dir should reach auth: {error}"))?;
    anyhow::ensure!(
        cli.work_dir.as_deref() == Some(std::path::Path::new("/tmp/plan")),
        "--work-dir did not reach the auth verbs"
    );

    // `--key-details` is optional: it defaults to the user and the host.
    let cli = parse(&[
        "auth",
        "register",
        "--user-preauth-key",
        "user_preauth_k1.secret",
    ])
    .map_err(|error| anyhow::anyhow!("register should parse without --key-details: {error}"))?;
    let Command::Auth(args) = cli.command else {
        anyhow::bail!("register should route to the auth verbs");
    };
    anyhow::ensure!(
        matches!(&args.command, AuthCommand::Register(register) if register.key_details.is_none()),
        "register should leave --key-details unset: {:?}",
        args.command
    );

    // Without an invite, register is a request for an account, and the address
    // an operator answers is the one thing it cannot derive.
    let cli = parse(&["auth", "register", "--contact-email", "ops@example.com"])
        .map_err(|error| anyhow::anyhow!("register should parse as a request: {error}"))?;
    let Command::Auth(args) = cli.command else {
        anyhow::bail!("register should route to the auth verbs");
    };
    anyhow::ensure!(
        matches!(&args.command, AuthCommand::Register(register)
            if register.user_preauth_key.is_none() && register.contact_email.is_some()),
        "register should carry the address and no invite: {:?}",
        args.command
    );

    // The flag takes mgmt's rule, and a refusal names it.
    let cli = parse(&["auth", "register", "--contact-email", "ops@münchen.de"])
        .map_err(|error| anyhow::anyhow!("register should take a non-ASCII domain: {error}"))?;
    let Command::Auth(args) = cli.command else {
        anyhow::bail!("register should route to the auth verbs");
    };
    anyhow::ensure!(
        matches!(&args.command, AuthCommand::Register(register)
            if register.contact_email.as_ref().map(|email| email.as_str()) == Some("ops@münchen.de")),
        "register should carry the address as typed: {:?}",
        args.command
    );
    let error = parse(&["auth", "register", "--contact-email", "ops@localhost"])
        .err()
        .ok_or_else(|| {
            anyhow::anyhow!("register should refuse an address with no dotted domain")
        })?;
    anyhow::ensure!(
        error.to_string().contains("dotted domain"),
        "clap error should name the rule: {error}"
    );

    // Neither flag is no path at all, and clap says which one is missing.
    let error = parse(&["auth", "register"])
        .err()
        .ok_or_else(|| anyhow::anyhow!("register should refuse with neither flag"))?;
    anyhow::ensure!(
        error.to_string().contains("--contact-email"),
        "clap error should name --contact-email: {error}"
    );

    // Both at once is refused at the flag: an invite already names the account.
    let error = parse(&[
        "auth",
        "register",
        "--user-preauth-key",
        "user_preauth_k1.secret",
        "--contact-email",
        "ops@example.com",
    ])
    .err()
    .ok_or_else(|| anyhow::anyhow!("register should refuse an invite with an address"))?;
    anyhow::ensure!(
        error.to_string().contains("cannot be used with"),
        "clap error should report the conflict: {error}"
    );

    // The key a registration signs with: named, or minted, never both. Nothing
    // names it by default, which is the machine-with-one-key case.
    let cli = parse(&[
        "auth",
        "register",
        "--contact-email",
        "ops@example.com",
        "--ssh-key",
        "SHA256:eJQWmJmce1mUjjLjicIm/xIjF6YmW2R9Aoz0r6yyFjo",
    ])
    .map_err(|error| anyhow::anyhow!("register should take --ssh-key: {error}"))?;
    let Command::Auth(args) = cli.command else {
        anyhow::bail!("register should route to the auth verbs");
    };
    anyhow::ensure!(
        matches!(&args.command, AuthCommand::Register(register)
            if register.key.ssh_key.is_some() && !register.key.generate),
        "register should carry the named key: {:?}",
        args.command
    );

    let error = parse(&[
        "auth",
        "adopt",
        "--user-id",
        "user_0192f0c1",
        "--ssh-key",
        "laptop",
        "--generate",
    ])
    .err()
    .ok_or_else(|| anyhow::anyhow!("adopt should refuse a named key and --generate at once"))?;
    anyhow::ensure!(
        error.to_string().contains("cannot be used with"),
        "clap error should report the conflict: {error}"
    );

    Ok(())
}
