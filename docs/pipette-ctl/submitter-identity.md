# Submitter identity

`pipette-ctl` talks to the management server (pipette-mgmt) as a **user**
(`user_…`), not as a client (`ev1_…`). This page covers the user-side calls
it makes and the local files they read and write. The management server
defines the two principal kinds, their keys, permissions and registration
rules: pipette-mgmt
[`docs/authentication.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md)
[§1](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#1-identity-model) and
[§5.11](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#511-users).

The client-side pieces are `crates/pipette-ctl/src/identity.rs` (the store,
shared with the management verbs) and `crates/pipette-ctl/src/auth.rs` (the verbs), over
`crates/pipette-mgmt-client/src/user_api.rs` (the calls). Each verb's flags and
defaults are in `pipette-ctl auth <verb> --help`.

## Where the identity lives

`<work_dir>/.pipette-ctl/identity/`, holding one file, or two:

| File | Contents |
|------|----------|
| `user.json` | `user_id`, `user_key_id`, `server_url`, and `ssh_key` when the account is registered against a key this machine already had |
| `id_ed25519` | the hex private key, created `0600`. Written only by `--generate` |

The management verbs ([management routes](management.md)) read the same
files, so an identity written by `auth register` is the identity a campaign is
filed under, without either naming a path.

`ssh_key` records the public half and where the key was found. The identity
finds the key again by its public half. One `ssh-add` moves a key between a
file and an agent, so an identity that recorded a socket or a path could
stop working after a change that has nothing to do with the account.

`.pipette-ctl` is `pipette-ctl`'s own workspace directory, the same one its
plan state uses, resolved through the same `storage_root` call. It is
deliberately not `.pipette`, which is the `pipette` CLI's tree and holds that
machine's *client* identity: the two principals are separate, so their key
material is kept separate too.

`--work-dir` selects which one, as it does for every other verb. There is no
second flag naming the identity directory: one would let `auth` write where
the management verbs do not read.

Every `auth` verb except `keys available` refuses a working directory with no
`.pipette-ctl` workspace in it, naming the `init` that creates one. The
identity is stored inside that workspace, so registering into an uninitialized
directory would leave key material beside no plan state.

## Logging in from the browser

If you can sign in to the management app, `auth login` approves this
machine's key from there, with no pre-auth key and no second machine:

```bash
pipette-ctl init                       # once, if the directory has no workspace
pipette-ctl auth login
```

It picks a key the way `auth register` does ([which key signs](#which-key-signs)),
prints a page at `<server>/app/login/cli` and a code, and opens the page in the
default browser. It opens nothing with `--no-browser`, in an SSH session, or on
Linux with neither `DISPLAY` nor `WAYLAND_DISPLAY` set; open the printed
address yourself then, on any machine.

The code is the first 8 hex digits of the key's SHA-256 (the digest in its
`userkey_` id), as `XXXX-XXXX`, and the page asks for it, so a page whose code
differs is approving another key. Eight digits catch a mix-up, not a page
built to match: approve only a page you opened from `auth login` yourself.
The management server defines what the page shows and the lifetimes it
offers (pipette-mgmt [`ui/README.md`, Signing in](https://github.com/Liquid4All/pipette-mgmt/blob/main/ui/README.md#signing-in);
the security model is
[`docs/authentication.md` §5.12](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#512-signing-in-from-a-browser)).

Meanwhile `auth login` asks `GET /api/v1/users/me` every 3 seconds, signed with
the key alone ([signing](#signing)). Once the key is approved it writes the
identity with the account the server names, as `auth register` would have, and
prints the account and the key. Nothing is written before that: Ctrl-C, or
`--timeout` passing (counted from when the page is printed), leaves the
directory as it was. A transport failure, `429` or `5xx` does not end the wait.

Run it again in a directory that already holds an identity to re-approve that
identity's key, after it expired or was revoked. If the key still works, it
says so and stops. `--ssh-key` and `--generate` are refused there: remove the
identity directory to log in with a different key, as the page suggests when
another account or a device holds this one.

A machine key that some account already holds needs no page either: `auth
login` writes the identity for that account and says so. If the directory was
stored for a different account, it says which one it rewrote it to.

## Getting an account

Two paths in, both through `auth register`. Which one you are on is decided by
what you pass: an invite joins an account that exists, an address asks for one.
The two flags are exclusive, refused at the flag rather than after a round
trip. How the server admits each, and what it refuses, is in pipette-mgmt
[`docs/authentication.md` §5.11](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#511-users)
(Provisioning and routes).

### With an invite (pre-approved)

An administrator creates the account (`pipette-ctl users create`, or
`pipette-mgmt users create` on the server host) and mints a **user pre-auth
key** for it, over HTTP (`pipette-ctl users preauth-create`) or on the host
(`pipette-mgmt users preauth create`); that key is what you present:

```bash
pipette-ctl init                       # once, if the directory has no workspace
pipette-ctl auth register \
  --user-preauth-key user_preauth_k1.secret
```

The registration it admits is active at once, so there is no approval step to
wait for. The key can also come from an environment variable (see `--help`),
which keeps it out of shell history; it is sent with the request and never
written to disk.

A user's pre-auth key is not a device's, so they are two types here,
`UserPreauthKey` and `ClientPreauthKey`, and the `client_preauth_...` token a
device registers with registers nothing here.

### Without one (requested)

No invite, no account yet. `--contact-email` asks for one:

```bash
pipette-ctl init
pipette-ctl auth register --contact-email ops@example.com
```

The account comes up `pending` and may do nothing until an operator admits it:
filing a campaign is refused, and `auth me` reports the status. The key is
chosen and stored exactly as on the invited path, so the account is signed for
the moment it is admitted and nothing here is re-run.

### Either way

Register refuses if this working directory already holds a user identity.
Overwriting would leave the registered key with no local copy, and revoking a
key has to be signed by a key the account holds.

It registers before writing anything, so a refused pre-auth key leaves no
identity on disk to clear before a retry.

## Which key signs

An account is registered against an Ed25519 public key, and a machine normally
has one already: in `~/.ssh`, in the agent of a vault like 1Password, or both.
`auth register`, `auth login` and `auth adopt` use that key rather than minting
a second one.

```bash
pipette-ctl auth keys available
```

lists what this machine offers. It reads every ssh-agent on the host, not only
the one `SSH_AUTH_SOCK` names, and every key under `~/.ssh`, merging a key
found in both places into one entry. Keys it cannot sign with are listed too,
because the reason is what an operator acts on: an encrypted file that no agent
holds needs an `ssh-add`, not a new key.

| What you pass | What happens |
|---|---|
| nothing | the machine's one usable key. Several, and it stops and lists them |
| `--ssh-key <selector>` | that key (selectors below) |
| `--generate` | a new key, written to the identity directory |

`--ssh-key` takes one of these selectors:

- a `SHA256:...` fingerprint, as `ssh-add -l` prints it;
- the comment on the key;
- its hex public key;
- the path of a private key file in `~/.ssh`.

Only keys in an agent or in `~/.ssh` can sign. For a key kept elsewhere,
`ssh-add` it and name it by fingerprint or comment.

Nothing is copied by registering a key the machine already has. A key an agent
holds is never read at all: each request is signed by the agent, which may
prompt (a vault's approval, a touch) and is given as long as it needs. A key in
an unencrypted file is read from where it already lives when the identity is
loaded, and copied nowhere.

`--ssh-key` and `--generate` are exclusive, and neither is implied: a machine
with no usable key is told what would make one usable rather than quietly
getting a second private key on disk.

On `auth register` and `auth login`, `--key-details` defaults to
`user@hostname`; `auth keys add` requires it. A key id is 64 hex characters, so
the description is what makes a later revocation decidable by a human, and the
operator and the machine are what tell two rows of `auth me` apart.

## Reading the account

```bash
pipette-ctl auth me          # GET /api/v1/users/me
pipette-ctl auth me --local  # from disk, no request
```

The server call prints the account, its type, status and organization, its
permission set, and the keys registered against it, with this working
directory's own key marked and each key's expiry. `--local` prints the stored
account id, key id, server and identity directory without a request. It still
loads the stored key first, so it fails when the key file or the system key it
names cannot be read.

## Adding a second machine

`POST /api/v1/users/me/keys` is signed by a key the account already holds, so a new
machine cannot register itself. The flow is two-sided.

On the new machine:

```bash
pipette-ctl init
pipette-ctl auth adopt --user-id user_0199b0f2-1d3c-7c41-9b6e-2a1f4c8d5e60
```

That picks this machine's key (or mints one with `--generate`), writes the
identity, and prints the public key. It contacts nothing. Until the next step
lands, requests from here are refused, because the account does not know the
key yet. `pipette-ctl auth pubkey` reprints the value if the output scrolled
away.

On a machine already registered to the account:

```bash
pipette-ctl auth keys add \
  --public-key <hex from the new machine> \
  --key-details "build server"
```

Or paste the public key into **Add a key** on the management app's Account
page (`<server>/app/account`), which needs no registered machine at all.
`auth login` on the new machine does both halves from the browser.

The key id is derived locally from the public key (`userkey_` plus its
SHA-256) before the request, so a malformed or truncated paste fails here
rather than at the server.

`--expires-in` makes the key stop signing after that many whole days; without
it the key never expires. The server sets the upper bound and refuses a
longer one (pipette-mgmt
[`docs/api/users.md`, Signing in from a browser](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/api/users.md#signing-in-from-a-browser)),
so a change to it needs no client release.

## Revoking a key

```bash
pipette-ctl auth keys revoke userkey_...   # DELETE /api/v1/users/me/keys/{key_id}
```

The server allows revoking the last key or the key signing the request
(pipette-mgmt [`docs/authentication.md` §5.11](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#511-users),
Keys), so a fresh user
pre-auth key is how a holder with no key left re-registers. Nothing local is
deleted, so remove `<work_dir>/.pipette-ctl/identity/` too when retiring the
key it holds.

## Signing in to the management app

The browser console is the app at `<server>/app/`. Its sign-in page offers
three ways in:

- Clerk;
- a private key file (the hex in `id_ed25519`, when `--generate` wrote one);
- **pipette-ctl on this machine**.

The last one makes a key in the browser and prints the second half of
[adding a second machine](#adding-a-second-machine) with it:

```bash
pipette-ctl auth keys add \
  --public-key <hex the page printed> \
  --key-details "<browser the page named>" \
  --expires-in 7d
```

Run it in the working directory whose identity you sign in as, then press
**I ran it, sign in**. Your own key never leaves this machine. The key you
added belongs to the browser and expires on its own, so a command that you
run and never follow up leaves nothing that lasts.

The page prints this command, so a change to the verb or its flags needs a
matching change in pipette-mgmt.

## Signing

A user signs every request with RFC 9421 HTTP Message Signatures, in the one
profile that the management server reads: pipette-mgmt
[`docs/authentication.md` §2.4](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#24-rfc-9421-signatures-user-accounts).
`crates/pipette-mgmt-client/src/httpsig.rs` builds the fields. It reproduces,
byte for byte, the test vector that the server's Rust and TypeScript signers
pin. Clients sign with the `v1` scheme
([§2.1](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#21-the-signed-payload)).

`keyid` names the key, and the key names the account, so no header carries the
account id. `UserIdentity` keeps the account id for local use.

`auth login` asks `GET /api/v1/users/me` before it knows the account, and signs
the same way: nothing else would name the account. The server answers `401 key
not found` until an account holds the key. `KeyOnlyIdentity` and
`user_api::me_by_key` make that request.

`UserIdentity` carries a `UserSigner`. A `UserSigner` is either the hex
private key or an outside signer that holds the key (`SignsPayload`).
`pipette-ctl` implements the outside signer over the ssh-agent client of
`pipette-ssh` (`crates/pipette-ctl/src/system_keys.rs`). Both produce the same
64 bytes for one signature base, so the server cannot tell them apart.

`pipette-mgmt-client` has one module for each principal: `client_api` for a
device and `user_api` for an account. The parameter type of each call
separates the two, so a client credential cannot reach a campaign route at
compile time. [Architecture](../architecture.md) describes the split.

## No client fallback

`pipette-ctl` signs only with its own user identity. It never signs with the
client identity in `.pipette/identity/`, and it has no `--client-id`, so an
account signs everything it files, through campaigns. One command reads that
directory: `catalog for-device` with no device id reads the device id from
`.pipette/identity/registration.json`, and still signs as the account
([management routes](management.md)). Only a user files a plan (pipette-mgmt
[`docs/authentication.md` §5.11](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#511-users),
Attribution;
[`docs/httpapi.md` §2.12](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md#212-post-apiv1plans)).

| CLI | Principal | Talks to the server about |
|---|---|---|
| `pipette` | client (`ev1_…`) | claiming jobs, submitting results, its own registration |
| `pipette-ctl` | user (`user_…`) | filing campaigns, administering the deployment, and its own account |
