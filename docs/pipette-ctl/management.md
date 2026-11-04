# Management routes: CLI verbs and MCP server

`pipette-ctl` reaches the routes of the management server (pipette-mgmt)
that an account can call. It leaves out the routes
[listed below](#left-out-on-purpose) and the browser console.

- It signs each request as the account stored in the working directory (see
  [submitter identity](submitter-identity.md)). The directory resolves as in
  [architecture: workspace](../architecture.md#workspace).
- It talks to the management server over HTTP only. It needs no shell on
  the server host and no `pipette-mgmt` binary.
- The server decides what the account may do
  ([pipette-mgmt docs/authentication.md §5](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#5-permissions)).
- [pipette-mgmt docs/httpapi.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md)
  and [`docs/api/`](https://github.com/Liquid4All/pipette-mgmt/tree/main/docs/api)
  describe each route.

There are two front ends, both generated from one table,
`crates/pipette-ctl/src/mgmt/catalog.rs`:

- **CLI verbs**: `pipette-ctl <area> <verb>`, for example
  `pipette-ctl orgs create --slug acme`.
- **MCP tools**: `pipette-ctl mcp serve` speaks MCP over stdio, with one tool
  per route named `<area>_<verb>` (`orgs_create`).

Each tool's input schema matches the verb's flags, because both come from the
same catalog row. `pipette-ctl mcp tools` lists every tool with its route.
The shared descriptions and errors name tools and arguments as MCP does; the
CLI's help spells them as its verbs, flags and positionals (`vault blob-put`,
`--overwrite`).

## MCP setup

Register the server with the working directory that holds the identity:

```sh
claude mcp add pipette -- pipette-ctl --work-dir ~/pipette mcp serve
```

An MCP client can remove the arguments, or read them as its own options
(the MCP Inspector does both). With such a client, name the directory
through the environment:

```sh
claude mcp add pipette -e PIPETTE_WORK_DIR="$HOME/pipette" -- pipette-ctl mcp serve
```

`--server` or `PIPETTE_MGMT_SERVER` points it at a different deployment. The
`initialize` instructions name the server and the account the session signs
as, with its key id, or say why there is no account; without one, only the
public catalog tools work. A working directory whose stored identity does not
resolve reports why on every signed call, and a stored server URL that cannot
be parsed stops `mcp serve` at startup rather than fall back to production.

`GET` and `HEAD` tools carry the `readOnlyHint` annotation, except
`vault_download`, which writes a local file. A tool carries
`destructiveHint` if a second call cannot undo its effect:

- a delete, revoke, reject or cancel;
- a replacement of a whole set;
- a campaign hand-off;
- a retype;
- a user pre-auth mint that replaces the account's other keys;
- `vault_download` over an existing file.

`pipette-ctl mcp tools --json` prints each tool's annotations.

A tool call runs beside the protocol, so the server answers `ping`,
`tools/list` and further calls while a call is in flight. Up to eight calls
run at once. The server does not answer a call that the client cancels. A
cancelled call that already reached the management server still takes
effect; one that still waits for a worker does not run.

Time limits:

- A call that gets no answer within two minutes fails.
- A download fails if it stalls for two minutes. It has no total limit.
- An upload has no time limit.

## Areas

| Area | Covers |
|------|--------|
| `clients` | the fleet: list, show, capability counts, update, approve, reject, unsuspend, set-owner, set-permissions, amend-permissions, delete |
| `users` | accounts: list, show, create, set-status, set-type, set-permissions, amend-permissions, set-org; an account's keys (keys, key-revoke) and pre-auth keys (preauth-keys, preauth-create, preauth-revoke) |
| `orgs` | organizations, their members and devices, assign-clients, private benchmarks, device pre-auth keys (client-preauth-keys, client-preauth-create, client-preauth-revoke), audit (the organization's log) |
| `audit` | list: the whole audit log, the host's commands and unverified callers' requests included |
| `campaigns` | list, create with a first plan, show, rename (and label), plan, plans, file a plan, document, revise, list-variants, variants, rerun, recalculate, set-owner, delete |
| `vault` | list, upload a model or runtime, blob-exists, blob-put, show, download, delete |
| `plans` | list, validate (check a document and file nothing), document, variants (its `[[variants]]` blocks as TOML), status, update (its note), jobs, results (as CSV), by-job (the plans that asked for a job, and its body), explain, cancel |
| `model-params` | the model catalog, each model's parameter counts: list, create, update, delete |
| `benchmarks` | the public catalog, readable without an identity |
| `catalog` | descriptor catalog: full list, named entry, and the members offered to a device |

The descriptor catalog commands are:

```sh
pipette-ctl catalog list --prefix llamacpp --kind stable
pipette-ctl catalog list --members --all
pipette-ctl catalog show global/llamacpp-v0.7.0
pipette-ctl catalog for-device ev1_example
pipette-ctl catalog for-device ev1_example --withheld
pipette-ctl catalog for-device --json
```

`list` prints entry and member summaries by default. `--members` adds each
full member and its `plan_toml` field. The server provides this field as a TOML
inline table, which a plan accepts as written. `show` and `for-device` always
print the `plan_toml` field of each member. The `for-device` command also
prints the run evidence, the withheld counts, and a summary of the reasons.
Pass `--withheld` to print each withheld member and its reason. `--json`
prints the full server answer, including withheld members, for scripts.

If `show` gets a name that the server does not know, it prints the entries that
it found. Then it stops with an error that names each unknown name. With
`--json`, `show` prints the server answer and exits with status 0. The answer
lists the unknown names in `unknown`.

Without a `client_id`, `for-device` reads the registration file of the pipette
client: `<work-dir>/.pipette/identity/registration.json`. The registration must
name the same server that the command uses. The command takes the server from
the user identity or from `--server`. To query a different device, pass its
client id. The server checks that your user has the permission to read that
device.

The account's own routes (`/api/v1/users/me`, key add and revoke) are MCP
tools too. Their CLI verbs are under `auth`, which also manages local keys. If
you revoke the key that the session signs with (`users_revoke_key`, or
`users key-revoke` / `users_key_revoke` on your own account), every later
call answers `401`.

**Work is filed only through campaigns.** `campaigns create` files the first
plan. `campaigns file`, `revise`, `variants`, `rerun` and `recalculate` file
each revision.

- The server expands each plan document and stores it. `variants`, `rerun`
  and `recalculate` read the stored document back. `campaigns document` reads
  it out, for `revise` to send back edited.
- `file` runs every cell again, and the plan it supersedes keeps running.
- The other four carry over the jobs they keep, and cancel the plan they
  replace.
- Each verb takes `dry_run`, which checks and expands the plan on the server
  and files nothing. `rerun --answered` and `recalculate --all-evals` have no
  dry run.

The server files no plan outside a campaign
([pipette-mgmt docs/httpapi.md §2.12](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md#212-post-apiv1plans),
[docs/plan-ingestion.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md)).

Device routes (claim, heartbeat, result submission, `/jobs/*`, `/datasets/*`)
admit only a client and are the `pipette` CLI's. The browser console, the app
at `<server>/app/`, is not covered; [submitter
identity](submitter-identity.md#signing-in-to-the-management-app) says how to
sign in to it.

### Left out on purpose

These account routes have no verb and no tool:

- `GET /api/v1/openapi.json`, the description of every route, is a document
  for tools to read, not an account action.
- `POST /api/v1/users/register` is `auth register`, which also stores the
  identity it registers. As a tool, it would spend a single-use pre-auth key
  and store nothing.
- `POST /api/v1/users/register/clerk` is a browser's: a Clerk token
  authenticates it, not a key. `auth login` is the browser path here.
- The app's page reads, `GET /api/v1/orgs/{org_id}/overview` and
  `GET /api/v1/campaigns/{campaign_id}/overview`, draw a page in one read from
  what the area verbs list.
- The app's owner-field lookups, `GET /api/v1/clients/{client_id}/owners` and
  `GET /api/v1/campaigns/{campaign_id}/owners`, offer the choices of a form.
  The `set-owner` verbs take any id, and the server judges it.
- `GET /api/v1/vault/types` is the upload form's table of types and their
  files, which `vault upload`'s help carries.
- `GET /api/v2/benchmarks` and `GET /api/v2/benchmarks/{benchmark_id}` serve
  the same catalog, each entry cut to the `benchmark_type` +
  `benchmark_params` a submission declares; `benchmarks list` and
  `benchmarks show` read the whole entries.

`POST /api/v1/campaigns/{campaign_id}/revert` has no verb and no tool either:
the catalog does not carry it.

## Arguments

These conventions hold for every verb and tool. What one verb takes and checks
(owner ids, plan and benchmark documents, labels, vault files, downloads) is in
its `--help` and its tool description, both written in the catalog row.

- Path parameters are positional on the CLI (`orgs show acme`). A value of `.`
  or `..`, or one spelled like a route beside it (`clients show capabilities`,
  `users show me`), is refused: the request would reach that route instead.
- Paged listings (`clients list`, `orgs audit`, `audit list`,
  `campaigns list`, `vault list`, `plans list`) take `--limit` and
  `--cursor`. `--all` reads every page from the first. It fails on an answer
  that is not a page; it does not read that answer as empty.
- `plans jobs` pages with `--limit` and `--offset`. Its `--all` is the
  server's own flag, which answers every job at once. `--results` adds each
  job's result, so `plans jobs --results` is the JSON read of the plan's
  results. The other listings answer in one response.
- A read that takes `--names` adds `names`, keyed by id: the owners'
  addresses or handles on the campaign and vault reads, the submitters'
  addresses on the plan reads; each flag's help says which ids. Under `--all`,
  every page's `names` merge into one.
- `--fields a,b` (MCP `fields`) keeps only those top-level keys of each row
  of a listing, or of the answer.
  - The server leaves out a key with no value, so a missing key comes back
    `null`, with a note that names the keys there are.
  - A record that holds none of the keys is refused. A listing is not,
    because a later page may hold them.
  - Only JSON reads take `--fields`, so a write never reports failure after
    it ran.
  - An `eval` benchmark carries its whole sample set under `samples`. Name
    `fields` without `samples` to leave the set out.
- An MCP argument given as `null` counts as left out. The exceptions are
  `org_id: null` on `users_set_org`, which clears membership, and
  `owner: null` on `clients_set_owner`, which detaches the device. Both are
  required, so leaving one out is refused, and the CLI spells the `null`
  `none` (`users set-org user_… --org-id none`).
- A boolean flag, `--all` and `--overwrite` included, reads bare as `true`. A
  value follows an `=` (`--dry-run=false`), so a positional after the flag
  stays a positional.
- A file argument (a plan, a variant edit, a benchmark definition, a vault
  file, a download's `out`) given as a relative path over MCP resolves against
  the directory `pipette-ctl mcp serve` was started in, not `--work-dir`, so an
  absolute path is safest. A download stages its bytes beside `out` and moves
  them onto it only once they match the record's digest.
- A model with an `auth_token` in a plan or a variant edit has special
  handling; see [job generation: gated models](job-generation.md#gated-models).
- `--json` prints the server's answer as JSON. Under `--all`, every page
  merges into one `items` envelope. Without `--json`:
  - A listing prints as a table, with a column for every key that any row
    has and each row on one line. The table cuts a long value unless it is
    an id or a digest; `--json` shows it whole.
  - Under the table, `unreadable` says what the server could not read: a
    count for a page, or the ids of an organization's benchmarks that it
    keeps but cannot read.
  - A record prints as `key: value` lines. A list of texts that span lines
    (the blocks of `campaigns list-variants`) prints as one block each under
    `# key[index]`.
  - A text answer (a plan's document, its variants or its results) prints as
    itself.

Over MCP, results are compact JSON. The tool saves a result longer than
40,000 characters under `<work_dir>/.pipette-ctl/mcp-results/`. That limit is
below the size at which Claude Code moves a result into a file of its own.
The tool then answers a summary:

- the absolute path and the size of the file;
- the row count and keys of a listing, or the keys of a record;
- a short preview;
- a hint that names the arguments of that tool that ask for less (`fields`,
  `limit`, its filters). A text answer (a
plan's document, its variants or its results) is saved as itself, a `.txt`
file, so a saved document can be passed back as a plan file. On Unix only the
user can read the directory and its files, and only the newest hundred are
kept. An answer that cannot be saved comes back cut at 40,000 characters with a
note saying why, since the call has run.

## Adding a route

Add one row to `ROUTES` in `crates/pipette-ctl/src/mgmt/catalog.rs`. Its CLI
verb and MCP tool follow from the row. The tests check that tool and argument
names stay unique and that clap accepts every generated verb.

- A listing that is not paged names its `rows` key.
- A new path parameter gets a `param_help` entry in the same file.
- A static management-server route that the catalog leaves out goes in
  `UNCATALOGUED` (`crates/pipette-ctl/src/mgmt/mod.rs`) when it sits where a
  catalog route takes a parameter. The CLI then refuses a value spelled like
  that route.
- Name a route left out on purpose under
  [Left out on purpose](#left-out-on-purpose).
