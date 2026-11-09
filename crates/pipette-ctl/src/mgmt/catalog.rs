//! The management routes an account can call, as data. mgmt files a plan only
//! into a campaign: `POST /api/v1/plans` answers `405`.
//!
//! One row per route drives a typed CLI verb (`pipette-ctl <area> <verb>`).
//! The rows mirror mgmt's
//! `GET /api/v1/openapi.json`; the server stays the judge of every value, so a
//! field here carries only what a caller needs to spell it.

use strum::VariantNames as _;

use pipette_mgmt_client::HttpMethod;
use pipette_plan_types::{ClientPermission, UserPermission};

use super::{ALL, FIELDS, OUT, OVERWRITE};

#[derive(Debug)]
pub struct Route {
    /// CLI group, e.g. `orgs`.
    pub area: &'static str,
    /// CLI verb, e.g. `create`.
    pub verb: &'static str,
    pub method: HttpMethod,
    /// Server path with `{name}` placeholders, each a required string argument.
    pub path: &'static str,
    /// First line is the CLI `about`; the rest follows.
    pub about: &'static str,
    pub query: &'static [Field],
    pub body: Body,
    pub response: Response,
    /// The route admits an unsigned request, so a working directory with no
    /// identity can still call it.
    pub public: bool,
    /// The key holding the records of a listing that is not a
    /// [`Response::Page`], e.g. `jobs` in `{"jobs": […]}`.
    pub rows: Option<&'static str>,
}

/// What a route sends.
#[derive(Debug)]
pub enum Body {
    None,
    /// A JSON object with these fields; absent optional fields are omitted.
    Fields(&'static [Field]),
    /// The one field's value is the whole body: JSON, or the TOML document a
    /// [`Kind::JsonOrToml`] was given as.
    Whole(&'static Field),
    /// The one field is a TOML document, sent as the `application/toml` body;
    /// an optional one left out is sent empty.
    Document(&'static Field),
    /// The one field names a file, streamed as the body. The route's `sha256`
    /// path parameter is the file's digest, computed when left out.
    Blob(&'static Field),
}

/// What a route answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Response {
    /// A JSON value, or nothing (`204`).
    Json,
    /// A text document, e.g. a plan's TOML, so no `fields` to keep.
    Text,
    /// The paginated envelope `{items, next_cursor, unreadable}`, plus `names`
    /// where asked. Takes `limit` and `cursor`, plus `all` to follow every page.
    Page,
    /// Bytes, written to the path in the `out` argument.
    Stream,
    /// A `HEAD`: `{"exists": bool}`, a `404` being `false`.
    Exists,
    /// A JSON value that the route reads from the progress stream of an
    /// import. The route files a plan document.
    Ingest,
}

#[derive(Debug)]
pub struct Field {
    pub name: &'static str,
    pub kind: Kind,
    pub required: bool,
    pub help: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub enum Kind {
    String,
    /// A string, or `none` or `null` sent as `null`, for a value the server
    /// clears only on an explicit `null` and not on a missing key.
    Clearable,
    /// A non-negative integer: every one the routes take is a count.
    Integer,
    Boolean,
    /// One of these spellings.
    Enum(&'static [&'static str]),
    /// A list of strings; repeat the CLI flag.
    Strings,
    /// A list of strings sent as one comma-separated query value, as mgmt refuses
    /// the key repeated; repeat the CLI flag or separate with commas.
    CommaList,
    /// A list, each one of these spellings; in a query, one comma-separated
    /// value, as mgmt's audit filters read it; on the CLI, repeat the flag or
    /// separate with commas.
    Enums(&'static [&'static str]),
    /// Top-level keys of an answer, at least one; comma-separated on the CLI.
    Keys,
    /// A JSON object of string values: `key=value` on the CLI, repeatable, or
    /// the object as JSON text (`{}` for none).
    Pairs,
    /// A JSON object, or JSON or TOML given as text or as the path of a file
    /// holding it (`@path` too on the CLI). JSON goes as JSON, TOML as the
    /// document itself.
    JsonOrToml,
    /// An owner: `user_…` or `org_…` for the `{type, …_id}` object the server
    /// takes, or the object itself. Where `detach` holds, `none` or `null`
    /// sends `null`, which the server reads as no owner.
    Owner {
        detach: bool,
    },
    /// A local file path.
    Path,
    /// TOML text, or a path to a TOML file.
    Toml,
    /// One file of a vault upload: a local path, hashed and uploaded under the
    /// upload's owner unless already there, then sent as `{filename, sha256}`;
    /// or that object itself, for a file already uploaded.
    VaultFile,
}

impl Route {
    /// The name catalog text and errors call the route by, e.g. `orgs_create`,
    /// respelled as its command in CLI help.
    pub fn tool(&self) -> String {
        format!("{}_{}", self.area, self.verb).replace('-', "_")
    }

    /// The `{name}` placeholders in the path, in order.
    pub fn path_params(&self) -> impl Iterator<Item = &'static str> {
        self.path
            .split('/')
            .filter_map(|segment| segment.strip_prefix('{')?.strip_suffix('}'))
    }

    /// Every argument but the path's: the query's, the body's, then
    /// [`Route::added_fields`].
    pub fn fields(&self) -> impl Iterator<Item = &'static Field> {
        self.query
            .iter()
            .chain(self.body_fields())
            .chain(self.added_fields())
    }

    /// The arguments the invoker adds by how the route answers: paging, a
    /// download's `out` and `overwrite`, a read's `fields`. Only `limit` and
    /// `cursor` reach the server.
    pub fn added_fields(&self) -> impl Iterator<Item = &'static Field> {
        let (added, all): (&'static [Field], _) = match self.response {
            Response::Page => (PAGE_QUERY, Some(&ALL_PAGES)),
            Response::Stream => (DOWNLOAD, None),
            Response::Json | Response::Text | Response::Exists | Response::Ingest => (&[], None),
        };
        let fields = self.takes_fields().then_some(&KEEP_FIELDS);
        added.iter().chain(all).chain(fields)
    }

    pub fn body_fields(&self) -> &'static [Field] {
        match self.body {
            Body::None => &[],
            Body::Fields(fields) => fields,
            Body::Whole(field) | Body::Document(field) | Body::Blob(field) => {
                std::slice::from_ref(field)
            }
        }
    }

    /// A path parameter the invoker fills in when the caller leaves it out.
    pub fn derives(&self, param: &str) -> bool {
        (matches!(self.body, Body::Blob(_)) && param == "sha256")
            || (self.area == "catalog" && self.verb == "for-device" && param == "client_id")
    }

    /// How the CLI spells the route.
    pub fn command(&self) -> String {
        format!("{} {}", self.area, self.verb)
    }

    /// Takes `fields`. Only a JSON read, since narrowing can fail, and a write
    /// must not report as an error a change it made.
    pub fn takes_fields(&self) -> bool {
        self.method == HttpMethod::Get && matches!(self.response, Response::Json | Response::Page)
    }

    /// The key of the answer's records, for a listing.
    pub fn rows_key(&self) -> Option<&'static str> {
        match self.response {
            Response::Page => Some("items"),
            Response::Json
            | Response::Text
            | Response::Stream
            | Response::Exists
            | Response::Ingest => self.rows,
        }
    }

    /// The first line of [`Route::about`], without its full stop.
    pub fn summary(&self) -> &'static str {
        self.about
            .lines()
            .next()
            .unwrap_or_default()
            .trim_end_matches('.')
    }
}

pub(crate) const fn req(name: &'static str, kind: Kind, help: &'static str) -> Field {
    Field {
        name,
        kind,
        required: true,
        help,
    }
}

pub(crate) const fn opt(name: &'static str, kind: Kind, help: &'static str) -> Field {
    Field {
        name,
        kind,
        required: false,
        help,
    }
}

/// `limit` and `cursor`, which every [`Response::Page`] route sends.
pub static PAGE_QUERY: &[Field] = &[
    opt(
        "limit",
        Kind::Integer,
        "Rows per page; the server's default is 50 and its maximum 200 unless configured \
         otherwise",
    ),
    opt(
        "cursor",
        Kind::String,
        "`next_cursor` from the previous page",
    ),
];
static ALL_PAGES: Field = opt(
    ALL,
    Kind::Boolean,
    "Follow next_cursor from the first page and return every row",
);
static KEEP_FIELDS: Field = opt(
    FIELDS,
    Kind::Keys,
    "Keep only these top-level keys of each row of a listing, or of the answer. A missing \
     key comes back null, with a note naming the keys there are; a record holding none of \
     them is refused",
);
static DOWNLOAD: &[Field] = &[
    req(
        OUT,
        Kind::Path,
        "Local file path to write the bytes to (absolute is safest: a relative one resolves \
         against this process's working directory)",
    ),
    opt(OVERWRITE, Kind::Boolean, "Replace `out` if it exists"),
];

/// A path parameter's help, by its name.
pub fn param_help(name: &str) -> &'static str {
    match name {
        "client_id" => "Device id (`ev1_…`)",
        "user_id" => "Account id (`user_…`)",
        "key_id" => "Key registration id (`userkey_…`)",
        "preauth_key_id" => {
            "Pre-auth key id: the listing's `key_id`, the 16 hex characters between the token's \
             prefix and its `.`"
        }
        "key" => {
            "Model catalog key: a model name as catalog lookup normalizes it \
             (`lfm3-1.7b`), or a vault resource id; a `model_params_list` row's key"
        }
        "org_id" => "Organization id (`org_…`) or handle",
        "principal_id" => "Owner of the file: a `user_…` or `org_…` id, not a handle",
        "sha256" => "The file's SHA-256: 64 hex characters, sent lowercased",
        "benchmark_id" => "Benchmark id, e.g. `decode_throughput_512_100`",
        "campaign_id" => "Campaign id (`campaign_…`)",
        "resource_id" => "Vault resource id (`vault_…`)",
        "plan_id" => "Plan id (`plan-…`)",
        "job_id" => "Job id (`job-…`)",
        _ => "Path parameter",
    }
}

const fn route(
    area: &'static str,
    verb: &'static str,
    method: HttpMethod,
    path: &'static str,
    about: &'static str,
) -> Route {
    Route {
        area,
        verb,
        method,
        path,
        about,
        query: &[],
        body: Body::None,
        response: Response::Json,
        public: false,
        rows: None,
    }
}

const GET: HttpMethod = HttpMethod::Get;
const POST: HttpMethod = HttpMethod::Post;
const PUT: HttpMethod = HttpMethod::Put;
const PATCH: HttpMethod = HttpMethod::Patch;
const DELETE: HttpMethod = HttpMethod::Delete;
const HEAD: HttpMethod = HttpMethod::Head;

const PERMISSIONS: &[&str] = UserPermission::VARIANTS;
const CLIENT_PERMISSIONS: &[&str] = ClientPermission::VARIANTS;
pub(crate) const CLIENT_PREAUTH_USAGES: &[&str] = &["single_use", "multi_use"];
pub(crate) const CLIENT_STATUSES: &[&str] = &["pending", "approved"];
pub(crate) const CATALOG_KINDS: &[&str] = &["stable", "daily", "hand-made"];
pub(crate) const CATALOG_STATES: &[&str] = &["pinned", "active", "hidden"];
pub(crate) const DESCRIPTOR_KINDS: &[&str] = &["runtime", "model"];
const CLIENT_STATUS: Kind = Kind::Enum(CLIENT_STATUSES);
pub(crate) const OWNER_FILTERS: &[&str] = &["mine", "org"];
const OWNER_FILTER: Field = opt(
    "owner",
    Kind::Enum(OWNER_FILTERS),
    "mine: only records the caller owns. org: only organization-owned records in reach, \
     narrowed to one organization by `org_id`",
);
const ORG_FILTER: Field = opt(
    "org_id",
    Kind::String,
    "Only records owned by this organization (`org_…`)",
);
/// `owner` on the fleet listing, where an organization's devices are `org_id`.
pub(crate) const FLEET_OWNERS: &[&str] = &["mine"];
pub(crate) const LIST_ORDERS: &[&str] = &["oldest", "newest"];
const ORDER: Field = opt(
    "order",
    Kind::Enum(LIST_ORDERS),
    "oldest first (the default) or newest first; a newest-first page reads the whole listing \
     on the server, so `all` reads the pages oldest first and turns the rows around. A \
     `cursor` holds only under the order that issued it",
);
// Answer keys stay unquoted here: the CLI would respell a quoted `names` or
// `plan` as its flag.
const OWNER_NAMES: Field = opt(
    "names",
    Kind::Boolean,
    "Add names, keyed by id: each owner's address, or an organization's handle",
);
const SUBMITTER_NAMES: Field = opt(
    "names",
    Kind::Boolean,
    "Add names, keyed by id: each submitter's address",
);
const CAMPAIGN_PLAN: Field = opt(
    "plan",
    Kind::Boolean,
    "Add a summary of each campaign's plan: some of the keys `plans_status` answers (status, \
     note, change, submitted_by, cancel_requested), valued alike, and its progress snapshot's \
     counts; none for a plan outside your plan scope or one whose manifest does not read",
);
const VAULT_OBJECT: Field = opt(
    "object",
    Kind::Boolean,
    "Add whether each file is still stored, from a stat of it: present, resized (replaced \
     behind the record, with its bytes) or missing. A listing's row the store did not answer \
     reads unknown; a read of one resource fails with 500",
);
pub(crate) const PLAN_STATUSES: &[&str] = &[
    "creating",
    "active",
    "pending_clients",
    "cancelling",
    "complete",
    "cancelled",
];
const PLAN_STATUS: Kind = Kind::Enum(PLAN_STATUSES);
pub(crate) const PLAN_SCOPES: &[&str] = &["mine", "all"];
pub(crate) const RERUN_ANSWERED: &[&str] = &["failed", "all"];
pub(crate) const USER_STATUSES: &[&str] = &["active", "pending", "disabled"];
pub(crate) const ACCOUNT_TYPES: &[&str] = &["human", "service"];
pub(crate) const ORG_STATUSES: &[&str] = &["active", "disabled"];
/// `VaultModel`'s tags, then `VaultRuntime`'s.
pub(crate) const VAULT_TYPES: &[&str] = &[
    "gguf_text",
    "gguf_vision",
    "gguf_vision_speculative",
    "mlx",
    "mlx_speculative",
    "torch",
    "torch_speculative",
    "openvino",
    "onnx_genai",
    "llamacpp_cli_stock_tools",
    "ovms",
    "trt_edgellm",
];
/// mgmt's `benchmark_type` tags, as `BenchmarkType` spells them.
pub(crate) const BENCHMARK_TYPES: &[&str] = &[
    "prefill_throughput",
    "decode_throughput",
    "end_to_end_latency",
    "max_memory_usage",
    "eval",
    "vlm_eval",
    "asr_eval",
    "vl_throughput",
    "vl_max_memory",
    "vl_spec_decode",
    "asr_throughput",
    "asr_max_memory",
];
const NOTE: Field = opt(
    "note",
    Kind::String,
    "Free text on why the plan is filed, at most 4096 characters once trimmed; \
     `plans_update` changes it later",
);
const DRY_RUN: Field = opt(
    "dry_run",
    Kind::Boolean,
    "Expand and validate on the server, and file nothing. Runs no fleet match: a plan \
     no working client can run still passes; only a real filing's `warnings` say so",
);
const PLAN_DOCUMENT: Field = req(
    "plan",
    Kind::Toml,
    "Scheduler-mode plan TOML text, or a path to a TOML file (absolute is safest: a \
     relative one resolves against this process's working directory). The server expands \
     and stores it",
);
const CHECKED_DOCUMENT: Field = req(
    "plan",
    Kind::Toml,
    "Scheduler-mode plan TOML text, or a path to a TOML file (absolute is safest: a \
     relative one resolves against this process's working directory). The server expands \
     and checks it, and stores nothing",
);
pub(crate) const AUDIT_OUTCOMES: &[&str] = &["succeeded", "refused", "failed"];
const AUDIT_OUTCOME: Field = opt(
    "outcome",
    Kind::Enums(AUDIT_OUTCOMES),
    "Only events with one of these outcomes; `refused` is a 401, a 403, or a 404 that hides \
     a record outside your scope. A list; on the CLI, repeat the flag or separate with commas",
);
pub(crate) const AUDIT_ACTORS: &[&str] = &["user", "client", "unverified", "host"];
const AUDIT_ACTOR: Field = opt(
    "actor",
    Kind::Enums(AUDIT_ACTORS),
    "Only events whose actor is of one of these kinds: an account, a device, a caller whose \
     signature did not verify, or a `pipette-mgmt` command on the host. A list; on the CLI, \
     repeat the flag or separate with commas",
);
const AUDIT_NAMES: Field = opt(
    "names",
    Kind::Boolean,
    "Add names, keyed by id: each id on the page that you may read by another route",
);
const PREAUTH_DAYS: Field = opt(
    "expires_in_days",
    Kind::Integer,
    "Days the key stands, 1 to 90; 90 when left out",
);
const TOTAL_MILLIONS: Field = req(
    "total_millions",
    Kind::Integer,
    "Total parameters, in millions; at least 1",
);
const VARIANT_EDIT: Field = opt(
    "edit",
    Kind::Toml,
    "Variant edit TOML text, or a path to a TOML file (absolute is safest: a relative one \
     resolves against this process's working directory): `remove = [indices]` and \
     `[[variants]]` to append. Left out, the edit is empty, which only `add_from` makes valid",
);
const CARRY_EARLIER: Field = opt(
    "carry_earlier",
    Kind::Boolean,
    "Let a cell the replaced plan has not answered take a finished run of the campaign's \
     earlier plans, as it takes one of that plan's. Files the plan's own document too",
);
const INHERIT_FROM: Field = opt(
    "inherit_from",
    Kind::CommaList,
    "Plans of any campaign, each one you can read (404 otherwise), whose finished runs a cell \
     the replaced plan has not answered may take, as it takes one of that plan's; at most 32 \
     with any `add_from`. Files the plan's own document too. A list; on the CLI, repeat the \
     flag or separate with commas",
);
const BLOB_FILE: Field = req("file", Kind::Path, "Path to the local file to upload");
const BENCHMARK_DEF: Field = req(
    "definition",
    Kind::JsonOrToml,
    "The benchmark definition: a JSON object tagged by `benchmark_type`, in the shape \
     `orgs_benchmark_show` answers, whose \"org_id\" and \"benchmark_id\" keys are dropped \
     since the path carries both (an `eval` read through `benchmarks_show` also carries \
     `samples`, which the server refuses: leave it out); or the TOML a catalog file under \
     `benchmarks/` holds, sent as it is. Either may be text or the path of a file holding it \
     (absolute is safest)",
);

pub static ROUTES: &[Route] = &[
    // ── the fleet ───────────────────────────────────────────────────────────
    Route {
        query: &[
            opt("status", CLIENT_STATUS, "Only clients in this status"),
            opt(
                "tag",
                Kind::Strings,
                "Only clients carrying every tag; repeatable",
            ),
            opt(
                "requires",
                Kind::Strings,
                "Capability flags every client must hold, whitespace-separated; repeatable",
            ),
            opt(
                "any_of",
                Kind::Strings,
                "A flag group of which a client must hold one; repeatable",
            ),
            opt(
                "excludes",
                Kind::Strings,
                "Flags a client must not hold; repeatable",
            ),
            opt(
                "not_all_of",
                Kind::Strings,
                "A flag group a client must not hold all of; repeatable",
            ),
            opt(
                "owned",
                Kind::Boolean,
                "true: only devices somebody owns. false: only devices nobody owns",
            ),
            // Not `OWNER_FILTER`: an `org` owner is a 400 here.
            opt(
                "owner",
                Kind::Enum(FLEET_OWNERS),
                "mine: only devices the caller owns as a person",
            ),
            opt(
                "org_id",
                Kind::String,
                "Only devices this organization (`org_…`) owns, directly or through a member. \
                 Gated, not only a filter: needs `read_orgs` over it (403 without, 404 outside \
                 your reach)",
            ),
        ],
        response: Response::Page,
        ..route(
            "clients",
            "list",
            GET,
            "/api/v1/clients",
            "List the devices in reach, with their capabilities, tags and permissions.\n\
             Needs `read_clients`; `read_any_clients` reaches the whole fleet. A capability \
             clause without `requires` matches nobody. `owned`, `owner` and `org_id` exclude \
             each other.",
        )
    },
    Route {
        query: &[opt("status", CLIENT_STATUS, "Only clients in this status")],
        ..route(
            "clients",
            "capabilities",
            GET,
            "/api/v1/clients/capabilities",
            "Count the devices holding each capability flag.\nNeeds `read_clients`.",
        )
    },
    route(
        "clients",
        "show",
        GET,
        "/api/v1/clients/{client_id}",
        "Read one device, with its capabilities, tags and permissions.\nNeeds `read_clients`.",
    ),
    Route {
        body: Body::Fields(&[
            opt(
                "organization",
                Kind::String,
                "Organization label on the record",
            ),
            opt(
                "client_details",
                Kind::String,
                "Free-form device description",
            ),
            opt(
                "contact_email",
                Kind::String,
                "Contact address for the device",
            ),
        ]),
        ..route(
            "clients",
            "update",
            PATCH,
            "/api/v1/clients/{client_id}",
            "Edit a device's descriptive fields; at least one is required.\n\
             Needs `manage_clients` over the device. A field can be replaced, not cleared.",
        )
    },
    route(
        "clients",
        "approve",
        POST,
        "/api/v1/clients/{client_id}/approve",
        "Approve a pending device so it may claim work.\n\
         Needs `manage_clients`. Idempotent.",
    ),
    route(
        "clients",
        "reject",
        POST,
        "/api/v1/clients/{client_id}/reject",
        "Reject a pending device, deleting its registration.\n\
         Needs `manage_clients`. 409 if the device is not pending.",
    ),
    route(
        "clients",
        "unsuspend",
        POST,
        "/api/v1/clients/{client_id}/unsuspend",
        "Lift a device's suspension so it claims work again.\nNeeds `manage_clients`.",
    ),
    Route {
        body: Body::Fields(&[req(
            "owner",
            Kind::Owner { detach: true },
            "New owner: a `user_…` or `org_…` id (an organization by its id, not its handle), \
             or the owner object; `none` or null detaches the device",
        )]),
        ..route(
            "clients",
            "set-owner",
            PUT,
            "/api/v1/clients/{client_id}/owner",
            "Hand a device to an organization or an account, or detach it.\n\
             Needs `manage_clients` over both the device and the destination.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "permissions",
            Kind::Enums(CLIENT_PERMISSIONS),
            "The complete permission set; repeatable, none clears it",
        )]),
        ..route(
            "clients",
            "set-permissions",
            PUT,
            "/api/v1/clients/{client_id}/permissions",
            "Replace a device's permission set.\n\
             Needs `manage_clients` over the device. An empty set leaves the device \
             registered and able to do nothing, and a stored name the server does not define \
             is dropped. `perform_work` lets the device claim any organization's queued work \
             it matches. Takes effect on the device's next request; `changed: false` wrote \
             nothing.",
        )
    },
    Route {
        body: Body::Fields(&[
            opt(
                "add",
                Kind::Enums(CLIENT_PERMISSIONS),
                "Permissions to grant; repeatable",
            ),
            opt(
                "remove",
                Kind::Enums(CLIENT_PERMISSIONS),
                "Permissions to revoke; repeatable",
            ),
        ]),
        ..route(
            "clients",
            "amend-permissions",
            PATCH,
            "/api/v1/clients/{client_id}/permissions",
            "Grant and revoke individual permissions of a device.\n\
             Needs `manage_clients` over the device. Give `add`, `remove` or both, with no \
             name in both. `perform_work` lets the device claim any organization's queued work \
             it matches. `changed: false` wrote nothing.",
        )
    },
    route(
        "clients",
        "delete",
        DELETE,
        "/api/v1/clients/{client_id}",
        "Delete a device's registration.\n\
         Needs `manage_clients` over the device. Idempotent: with `manage_any_clients` an \
         unknown id also answers success, so check the id with `clients_show` first.",
    ),
    // ── accounts ────────────────────────────────────────────────────────────
    Route {
        rows: Some("items"),
        ..route(
            "users",
            "list",
            GET,
            "/api/v1/users",
            "List the accounts in reach.\nNeeds `read_orgs`; `read_any_orgs` reaches every account.",
        )
    },
    Route {
        body: Body::Fields(&[
            req(
                "contact_email",
                Kind::String,
                "The account's contact address",
            ),
            opt(
                "permissions",
                Kind::Enums(PERMISSIONS),
                "Permissions to grant, capped at the caller's own; repeatable",
            ),
            opt(
                "org_id",
                Kind::String,
                "Organization id (`org_…`). A `manage_orgs` caller may name only its own and \
                 defaults to it; a `manage_any_orgs` caller's account joins none unless named",
            ),
            opt(
                "account_type",
                Kind::Enum(ACCOUNT_TYPES),
                "human, the default, or service for a job's account: it belongs to an \
                 organization (a `manage_any_orgs` caller names `org_id`) and holds no \
                 `manage_orgs`, `manage_clients` or `_any_` name",
            ),
        ]),
        ..route(
            "users",
            "create",
            POST,
            "/api/v1/users",
            "Create an active account with no key.\n\
             Needs `manage_orgs`. Its holder signs in by registering with a user pre-auth key \
             minted for it, by its administrator with `users_preauth_create` or on the server \
             host (`pipette-mgmt users preauth create`). Registering by the address instead is \
             refused: 409, or 403 where the server requires a pre-auth key or a Clerk rule \
             covers the address.",
        )
    },
    route(
        "users",
        "show",
        GET,
        "/api/v1/users/{user_id}",
        "Read one account with its resolved permissions.\nNeeds `read_orgs`.",
    ),
    Route {
        body: Body::Fields(&[req(
            "status",
            Kind::Enum(USER_STATUSES),
            "The account's new status",
        )]),
        ..route(
            "users",
            "set-status",
            PATCH,
            "/api/v1/users/{user_id}",
            "Approve, disable or re-enable an account.\n\
             Needs `manage_orgs`; approving one asked for by address takes `manage_any_orgs`, \
             since it belongs to no organization. An account cannot disable itself.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "account_type",
            Kind::Enum(ACCOUNT_TYPES),
            "The account's new type: human, or service for a job's",
        )]),
        ..route(
            "users",
            "set-type",
            PATCH,
            "/api/v1/users/{user_id}",
            "Retype an account as a person's or a job's.\n\
             Needs `manage_orgs`. A service account belongs to an organization (400 \
             otherwise) and holds no `manage_orgs`, `manage_clients` or `_any_` name (409 \
             otherwise), and a person's account becomes one only while nobody has signed in \
             as it (409 otherwise; the host's `pipette-mgmt users set-type` covers the rest). \
             Making a service a person's is always open. `changed: false` wrote nothing.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "permissions",
            Kind::Enums(PERMISSIONS),
            "The complete permission set; repeatable, none clears it",
        )]),
        ..route(
            "users",
            "set-permissions",
            PUT,
            "/api/v1/users/{user_id}/permissions",
            "Replace an account's permission set.\n\
             Needs `manage_orgs`; grants are capped at the caller's own.",
        )
    },
    Route {
        body: Body::Fields(&[
            opt(
                "add",
                Kind::Enums(PERMISSIONS),
                "Permissions to grant; repeatable",
            ),
            opt(
                "remove",
                Kind::Enums(PERMISSIONS),
                "Permissions to revoke; repeatable",
            ),
        ]),
        ..route(
            "users",
            "amend-permissions",
            PATCH,
            "/api/v1/users/{user_id}/permissions",
            "Grant and revoke individual permissions.\n\
             Needs `manage_orgs`; grants are capped at the caller's own. Removing a name that \
             a held wider name implies changes nothing: remove the wider one.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "org_id",
            Kind::Clearable,
            "Organization id (`org_…`, not a handle); `none` or null clears membership",
        )]),
        ..route(
            "users",
            "set-org",
            PUT,
            "/api/v1/users/{user_id}/org",
            "Move an account into an organization, or out of every one.\nNeeds `manage_orgs`.",
        )
    },
    Route {
        rows: Some("items"),
        ..route(
            "users",
            "keys",
            GET,
            "/api/v1/users/{user_id}/keys",
            "List the keys an account signs in with, oldest first.\n\
             Needs `manage_orgs` over the account. `auth me` lists your own too.",
        )
    },
    route(
        "users",
        "key-revoke",
        DELETE,
        "/api/v1/users/{user_id}/keys/{key_id}",
        "Revoke one of an account's keys.\n\
         Needs `manage_orgs` over the account. The key stops signing on the next request, and \
         the account's other keys go on signing. To retire a stolen key, revoke it and every \
         key added since (`users_keys` lists them oldest first) before the account is enabled \
         again. Your own account's keys revoke here too, as through `auth keys revoke`, the \
         one this session signs with included, which locks it out.",
    ),
    Route {
        rows: Some("items"),
        ..route(
            "users",
            "preauth-keys",
            GET,
            "/api/v1/users/{user_id}/preauth-keys",
            "List an account's pre-auth keys, newest first, without their secrets.\n\
             Needs `manage_orgs` over the account. A spent key is gone and an expired one \
             stays, marked `expired`. A key the host minted has no `minted_by`.",
        )
    },
    Route {
        body: Body::Fields(&[
            PREAUTH_DAYS,
            opt(
                "note",
                Kind::String,
                "Who the key is for, as the listing shows it; at most 4096 characters",
            ),
        ]),
        ..route(
            "users",
            "preauth-create",
            POST,
            "/api/v1/users/{user_id}/preauth-keys",
            "Mint the single-use pre-auth key that registers an account's first machine.\n\
             Needs `manage_orgs` over the account and every name it holds. Only for an \
             account nobody has signed in as yet (`awaits_sign_in` on `users_show`), or a \
             service account: 409 for a person's account that holds a key or that somebody \
             has signed in as. It replaces any other key minted here for the account. `token` \
             is shown this once; its holder runs \
             `pipette-ctl auth register --user-preauth-key <token>`.",
        )
    },
    route(
        "users",
        "preauth-revoke",
        DELETE,
        "/api/v1/users/{user_id}/preauth-keys/{preauth_key_id}",
        "Revoke an account's pre-auth key, so it registers nothing more.\n\
         Needs `manage_orgs` over the account. 404 for a key that names another account.",
    ),
    // ── organizations ───────────────────────────────────────────────────────
    Route {
        rows: Some("items"),
        query: &[opt(
            "counts",
            Kind::Boolean,
            "false leaves out each row's `member_count` and `client_count`, sparing the server \
             a read of every membership and ownership; true by default",
        )],
        ..route(
            "orgs",
            "list",
            GET,
            "/api/v1/orgs",
            "List the organizations in reach, with member and device counts.\nNeeds `read_orgs`.",
        )
    },
    Route {
        body: Body::Fields(&[
            req(
                "slug",
                Kind::String,
                "Handle: `[a-z0-9_-]`, at most 64, not `org_…`",
            ),
            opt(
                "display_name",
                Kind::String,
                "Display name; defaults to the handle",
            ),
        ]),
        ..route(
            "orgs",
            "create",
            POST,
            "/api/v1/orgs",
            "Create an organization.\nNeeds `manage_any_orgs`.",
        )
    },
    route(
        "orgs",
        "show",
        GET,
        "/api/v1/orgs/{org_id}",
        "Read one organization.\nNeeds `read_orgs`.",
    ),
    Route {
        body: Body::Fields(&[
            opt("display_name", Kind::String, "New display name"),
            opt(
                "status",
                Kind::Enum(ORG_STATUSES),
                "New status; changing it takes `manage_any_orgs`",
            ),
        ]),
        ..route(
            "orgs",
            "update",
            PATCH,
            "/api/v1/orgs/{org_id}",
            "Rename an organization, or disable or re-enable it.\n\
             Needs `manage_orgs`; changing `status` takes `manage_any_orgs`. Give \
             `display_name`, `status` or both. `changed: false` wrote nothing.",
        )
    },
    route(
        "orgs",
        "delete",
        DELETE,
        "/api/v1/orgs/{org_id}",
        "Delete an organization.\n\
         Needs `manage_any_orgs`. 409 while an account belongs to it, a device is owned by it \
         directly, or it keeps a private benchmark, a campaign, a vault resource or an \
         unexpired client pre-auth key; the refusal names the counts. A device a member owns \
         does not block it.",
    ),
    Route {
        rows: Some("items"),
        ..route(
            "orgs",
            "members",
            GET,
            "/api/v1/orgs/{org_id}/users",
            "List an organization's accounts.\nNeeds `read_orgs`.",
        )
    },
    Route {
        rows: Some("items"),
        ..route(
            "orgs",
            "clients",
            GET,
            "/api/v1/orgs/{org_id}/clients",
            "List the devices an organization owns, directly or through a member.\n\
             Needs `read_orgs` and `read_clients`.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "client_ids",
            Kind::Strings,
            "Client ids (`ev1_…`) to hand over; repeatable",
        )]),
        ..route(
            "orgs",
            "assign-clients",
            PUT,
            "/api/v1/orgs/{org_id}/clients",
            "Hand a set of devices to an organization in one call.\n\
             Needs `manage_clients` over every device.",
        )
    },
    Route {
        rows: Some("items"),
        ..route(
            "orgs",
            "client-preauth-keys",
            GET,
            "/api/v1/orgs/{org_id}/client-preauth-keys",
            "List the pre-auth keys that register devices into an organization, newest first, \
             without their secrets.\n\
             Needs `manage_clients` for your own organization, or `manage_any_clients` for any; \
             another reads as absent (404). The host's keys naming the organization are listed \
             too, with what they seed. A spent single-use key is gone and an expired one stays, \
             marked `expired`.",
        )
    },
    Route {
        body: Body::Fields(&[
            PREAUTH_DAYS,
            opt(
                "usage",
                Kind::Enum(CLIENT_PREAUTH_USAGES),
                "multi_use for one key a rack shares; single_use when left out",
            ),
            opt(
                "note",
                Kind::String,
                "What the key is for, as the listing shows it; at most 4096 characters",
            ),
        ]),
        ..route(
            "orgs",
            "client-preauth-create",
            POST,
            "/api/v1/orgs/{org_id}/client-preauth-keys",
            "Mint a pre-auth key that registers devices approved and owned by an organization.\n\
             Needs what `orgs_client_preauth_keys` needs. The key carries no tags and no \
             permission template, so a device registers holding `submit_results`. `token` is \
             shown this once; a device presents it to `pipette auth register --preauth-key`. \
             A mint replaces no other key.",
        )
    },
    route(
        "orgs",
        "client-preauth-revoke",
        DELETE,
        "/api/v1/orgs/{org_id}/client-preauth-keys/{preauth_key_id}",
        "Revoke one of an organization's device pre-auth keys, so it registers nothing more.\n\
         Needs what `orgs_client_preauth_keys` needs. 404 for a key that names another \
         organization.",
    ),
    Route {
        query: &[
            AUDIT_OUTCOME,
            AUDIT_ACTOR,
            opt(
                "target",
                Kind::String,
                "Only events naming this record by id, as a target or as the account or device \
                 that acted; with `manage_orgs` alone, another organization's id is refused (400)",
            ),
            AUDIT_NAMES,
        ],
        response: Response::Page,
        ..route(
            "orgs",
            "audit",
            GET,
            "/api/v1/orgs/{org_id}/audit",
            "List what changed in an organization, and who changed it, newest first.\n\
             Needs `manage_orgs` for your own organization, or `manage_any_orgs` for any; \
             another reads as absent (404). With `manage_orgs` alone, the events name no other \
             organization. Each write request and `pipette-mgmt` command that changed records is \
             an event, refused ones included. 404 too when the server keeps no audit log.",
        )
    },
    Route {
        rows: Some("benchmarks"),
        ..route(
            "orgs",
            "benchmarks",
            GET,
            "/api/v1/orgs/{org_id}/benchmarks",
            "List an organization's private benchmarks.\n\
             Members may read; `read_any_orgs` reaches all. `unreadable_benchmark_ids` names \
             the ids whose entry the server cannot read: served to nobody, each still blocks the \
             organization's delete until `orgs_benchmark_put` replaces it or \
             `orgs_benchmark_delete` removes it.",
        )
    },
    route(
        "orgs",
        "benchmark-show",
        GET,
        "/api/v1/orgs/{org_id}/benchmarks/{benchmark_id}",
        "Read one of an organization's private benchmarks.",
    ),
    Route {
        body: Body::Whole(&BENCHMARK_DEF),
        ..route(
            "orgs",
            "benchmark-put",
            PUT,
            "/api/v1/orgs/{org_id}/benchmarks/{benchmark_id}",
            "Create or replace a private benchmark.\n\
             Needs `manage_orgs` as a member, or `manage_any_orgs`. The definition is JSON, \
             or the TOML of a catalog file, which moves between the server host, the console \
             and this route unchanged. 409 if the public catalog or another organization \
             already holds the id, or another organization's entry pins its dataset and no \
             public one does.",
        )
    },
    route(
        "orgs",
        "benchmark-delete",
        DELETE,
        "/api/v1/orgs/{org_id}/benchmarks/{benchmark_id}",
        "Delete a private benchmark.\nNeeds `manage_orgs` as a member, or `manage_any_orgs`.",
    ),
    // ── the audit log ───────────────────────────────────────────────────────
    Route {
        query: &[
            AUDIT_OUTCOME,
            AUDIT_ACTOR,
            opt(
                "target",
                Kind::String,
                "Only events naming this record by id, as a target or as the account or device \
                 that acted: a `user_…`, `org_…`, `ev1_…`, `campaign_…`, `plan-…`, `job-…` or \
                 `vault_…` id",
            ),
            AUDIT_NAMES,
        ],
        response: Response::Page,
        ..route(
            "audit",
            "list",
            GET,
            "/api/v1/audit",
            "List every audit event, newest first.\n\
             Needs `manage_any_orgs`. Each write request and `pipette-mgmt` command that changed \
             records is an event, refused ones included; a write that changed nothing \
             (`changed: false`, a dry run) is none. Holds what no organization's log does: \
             unverified callers' requests, and host commands that name no organization or its \
             records. 404 when the server keeps no audit log.",
        )
    },
    // ── campaigns ───────────────────────────────────────────────────────────
    Route {
        query: &[
            OWNER_FILTER,
            ORG_FILTER,
            opt(
                "plan_id",
                Kind::String,
                "Only the campaign this plan was filed into, current or superseded: one page, so \
                 no `cursor`",
            ),
            ORDER,
            OWNER_NAMES,
            CAMPAIGN_PLAN,
        ],
        response: Response::Page,
        ..route(
            "campaigns",
            "list",
            GET,
            "/api/v1/campaigns",
            "List the campaigns in reach.\nNeeds `read_plans` or `manage_orgs`.",
        )
    },
    Route {
        query: &[
            req("name", Kind::String, "Campaign name"),
            opt(
                "owner",
                Kind::String,
                "Owner: a `user_…` or `org_…` id; defaults to the caller",
            ),
            NOTE,
            DRY_RUN,
        ],
        response: Response::Ingest,
        body: Body::Document(&PLAN_DOCUMENT),
        ..route(
            "campaigns",
            "create",
            POST,
            "/api/v1/campaigns",
            "Create a campaign with its first plan.\n\
             Needs `submit_plans`. The server expands and stores the plan document.",
        )
    },
    Route {
        query: &[OWNER_NAMES, CAMPAIGN_PLAN],
        ..route(
            "campaigns",
            "show",
            GET,
            "/api/v1/campaigns/{campaign_id}",
            "Read one campaign record: its name, labels, owner and `current_plan_id`.\n\
             `campaigns_plan` reads that plan's status.",
        )
    },
    Route {
        body: Body::Fields(&[
            opt("name", Kind::String, "New name"),
            opt(
                "labels",
                Kind::Pairs,
                "The whole set of labels, an object of key to value: a key left out is dropped, \
                 and `{}` clears them. At most 50, on the rules of an AWS tag; nothing on the \
                 server reads them. On the CLI, `key=value` repeatable, or the object as JSON",
            ),
        ]),
        ..route(
            "campaigns",
            "rename",
            PATCH,
            "/api/v1/campaigns/{campaign_id}",
            "Rename or label a campaign.\n\
             Needs to administer it: `submit_plans` for your own, `manage_orgs` for your \
             organization's or a disabled member's, or `manage_any_orgs` for any. Give `name`, \
             `labels` or both. `changed: false` wrote nothing.",
        )
    },
    Route {
        query: &[SUBMITTER_NAMES],
        ..route(
            "campaigns",
            "plan",
            GET,
            "/api/v1/campaigns/{campaign_id}/plan",
            "Read the status of the plan a campaign currently points at.\n\
             Needs `read_plans` as well as the campaign: a `manage_orgs` holder without it \
             reads the campaign and is refused its plan.",
        )
    },
    route(
        "campaigns",
        "plans",
        GET,
        "/api/v1/campaigns/{campaign_id}/plans",
        "List a campaign's current plan and the ones it superseded.",
    ),
    Route {
        query: &[NOTE, DRY_RUN],
        response: Response::Ingest,
        body: Body::Document(&PLAN_DOCUMENT),
        ..route(
            "campaigns",
            "file",
            POST,
            "/api/v1/campaigns/{campaign_id}/plans",
            "File a plan document under a campaign and point the campaign at it.\n\
             Needs `submit_plans` and the right to file into the campaign: your own, your \
             organization's as any member, or any with `manage_any_orgs`. Finished cells run \
             again and the plan it supersedes keeps running: to edit the plan a campaign \
             runs, keeping finished jobs, use `campaigns_revise`.",
        )
    },
    route(
        "campaigns",
        "document",
        GET,
        "/api/v1/campaigns/{campaign_id}/document",
        "Read the plan document a campaign runs, to edit and send back with `campaigns_revise`.\n\
         Needs what `campaigns_variants` needs. `base.plan_id` is the `base_plan_id` the \
         revision names. 409 if the campaign runs no plan, or one filed as job bodies.",
    ),
    Route {
        query: &[
            req(
                "base_plan_id",
                Kind::String,
                "The plan the document was read off: `base.plan_id` of `campaigns_document`",
            ),
            NOTE,
            DRY_RUN,
            CARRY_EARLIER,
            INHERIT_FROM,
        ],
        response: Response::Ingest,
        body: Body::Document(&PLAN_DOCUMENT),
        ..route(
            "campaigns",
            "revise",
            PUT,
            "/api/v1/campaigns/{campaign_id}/document",
            "Replace a campaign's whole plan document, then cancel the plan it replaces.\n\
             Needs what `campaigns_variants` needs. Jobs the document still describes are \
             carried over and not run again, finished ones included; `carry_earlier` and \
             `inherit_from` let a cell not yet answered take a finished run of another plan. \
             For what `campaigns_variants` cannot change: `benchmarks`, `expires_at`, \
             `merge_duplicates`, one field of a variant. The unchanged document files nothing \
             and answers `changed: false` unless `carry_earlier` or `inherit_from` is given, \
             and so does a retry whose first attempt landed. 409 if the campaign no longer \
             runs `base_plan_id`.",
        )
    },
    route(
        "campaigns",
        "list-variants",
        GET,
        "/api/v1/campaigns/{campaign_id}/variants",
        "Read the `[[variants]]` blocks of the plan a campaign runs, as a variant edit numbers \
         them.\n\
         Needs what `campaigns_variants` needs. `base.plan_id` is the `base_plan_id` the edit \
         names, and each block is its TOML text, at the index from 0 that its `remove` takes. \
         409 if the campaign runs no plan, or one filed as job bodies.",
    ),
    Route {
        query: &[
            req(
                "base_plan_id",
                Kind::String,
                "The plan the `remove` indices were read off: `base.plan_id` of \
                 `campaigns_list_variants`",
            ),
            NOTE,
            DRY_RUN,
            CARRY_EARLIER,
            INHERIT_FROM,
            opt(
                "add_from",
                Kind::CommaList,
                "Plans of any campaign, each one you can read (404 otherwise), whose variants \
                 are appended after the edit's, each with its plan's `benchmarks` where it names \
                 none; their finished runs are offered as `inherit_from`'s. A list; on the CLI, \
                 repeat the flag or separate with commas",
            ),
        ],
        response: Response::Ingest,
        body: Body::Document(&VARIANT_EDIT),
        ..route(
            "campaigns",
            "variants",
            POST,
            "/api/v1/campaigns/{campaign_id}/variants",
            "Revise a campaign's plan by its variants, then cancel the plan it replaces.\n\
             Needs what `campaigns_file` needs, plus `read_plans` and `cancel_plans`, whoever \
             filed the plan it runs; `read_any_plans` and `cancel_any_plans` instead when that \
             plan was filed outside the campaign or only `manage_any_orgs` reaches the \
             campaign; and `read_clients` or `read_any_clients`, since the document names \
             devices. Carried jobs are not run again; only added cells are new, and \
             `carry_earlier`, `inherit_from` and `add_from` let a cell not yet answered take a \
             finished run of another plan. 409 if the campaign no longer runs `base_plan_id`.",
        )
    },
    Route {
        query: &[
            opt(
                "base_plan_id",
                Kind::String,
                "The plan the campaign runs; required with `answered`",
            ),
            DRY_RUN,
        ],
        body: Body::Fields(&[
            opt(
                "job_ids",
                Kind::Strings,
                "Finished or failed jobs of the campaign's plan to run again; repeatable",
            ),
            opt(
                "answered",
                Kind::Enum(RERUN_ANSWERED),
                "Run again every job of the campaign's plan that failed, or every one with a \
                 result",
            ),
            NOTE,
        ]),
        ..route(
            "campaigns",
            "rerun",
            POST,
            "/api/v1/campaigns/{campaign_id}/rerun",
            "Run answered jobs of a campaign's plan again, then cancel the plan it replaces.\n\
             Needs what `campaigns_variants` needs. Give exactly one of `job_ids` or \
             `answered`; `answered` needs `base_plan_id` and has no dry run. Every other job \
             is carried over. The old results stay readable.",
        )
    },
    Route {
        query: &[
            opt(
                "base_plan_id",
                Kind::String,
                "The plan the campaign runs; required with `all_evals`",
            ),
            DRY_RUN,
        ],
        body: Body::Fields(&[
            opt(
                "job_ids",
                Kind::Strings,
                "Finished evals of the campaign's plan to score again; repeatable",
            ),
            opt(
                "all_evals",
                Kind::Boolean,
                "Score every finished eval of the campaign's plan again",
            ),
            NOTE,
        ]),
        ..route(
            "campaigns",
            "recalculate",
            POST,
            "/api/v1/campaigns/{campaign_id}/recalculate",
            "Score finished evals of a campaign's plan again, then cancel the plan it replaces.\n\
             Needs what `campaigns_variants` needs. No device runs them again. Give exactly \
             one of `job_ids` or `all_evals`; `all_evals` needs `base_plan_id` and has no dry \
             run. Every other job is carried over.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "owner",
            Kind::Owner { detach: false },
            "New owner: a `user_…` or `org_…` id (an organization by its id, not its handle), \
             or the owner object. A campaign always has one",
        )]),
        ..route(
            "campaigns",
            "set-owner",
            PUT,
            "/api/v1/campaigns/{campaign_id}/owner",
            "Hand a campaign to an organization or an account.\n\
             Needs what `campaigns_rename` needs over the campaign, and a destination you may \
             hand it to: yourself, the organization you belong to, a member of the organization \
             you administer, or anywhere with `manage_any_orgs`. An account taking it must hold \
             `submit_plans` (400 otherwise). Its plans move with it: from then on they are read \
             and cancelled under the new owner. A hand-off can leave you unable to administer \
             it: an organization's campaign is its `manage_orgs` holders', an account's is \
             that account's, and a `manage_any_orgs` holder administers every campaign.",
        )
    },
    route(
        "campaigns",
        "delete",
        DELETE,
        "/api/v1/campaigns/{campaign_id}",
        "Delete a campaign, keeping the plans it filed.\n\
         Needs what `campaigns_rename` needs. 409 while a plan it filed is still \
         `creating`, `active` or `pending_clients` with no cancel requested; the refusal names \
         each one, so cancel them first with `plans_cancel`. Once it is deleted, only \
         `read_any_plans` and `cancel_any_plans` reach its plans.",
    ),
    // ── the vault ───────────────────────────────────────────────────────────
    Route {
        query: &[OWNER_FILTER, ORG_FILTER, ORDER, OWNER_NAMES, VAULT_OBJECT],
        response: Response::Page,
        ..route(
            "vault",
            "list",
            GET,
            "/api/v1/vault/resources",
            "List the private artifacts in reach.",
        )
    },
    Route {
        body: Body::Fields(&[
            req("name", Kind::String, "Display name"),
            req(
                "type",
                Kind::Enum(VAULT_TYPES),
                "The plan variant type the files make up",
            ),
            opt(
                "owner",
                Kind::String,
                "Owner: a `user_…` or `org_…` id, not a handle; defaults to the caller",
            ),
            opt(
                "flavor",
                Kind::String,
                "What a prebuilt runtime was built for, as a plan spells it (e.g. `macos-arm64`, \
                 `windows-x64`, or a custom build's name). Required for \
                 `llamacpp_cli_stock_tools`, `ovms` and `trt_edgellm`, refused for every model \
                 type",
            ),
            opt("model", Kind::VaultFile, "A `.gguf` model file"),
            opt("mmproj", Kind::VaultFile, "A `.gguf` vision projector"),
            opt("draft", Kind::VaultFile, "A `.gguf` drafter"),
            opt(
                "dir_archive",
                Kind::VaultFile,
                "The model directory, packed as tar, tar.gz or zip",
            ),
            opt(
                "target_dir_archive",
                Kind::VaultFile,
                "The target model directory, packed",
            ),
            opt(
                "draft_dir_archive",
                Kind::VaultFile,
                "The drafter directory, packed",
            ),
            opt(
                "archive",
                Kind::VaultFile,
                "A prebuilt runtime's release archive",
            ),
        ]),
        ..route(
            "vault",
            "upload",
            POST,
            "/api/v1/vault/resources",
            "Upload a model's or prebuilt runtime's files and get the plan entry naming them.\n\
             Each file is a local path, uploaded unless already there, or `{filename, sha256}` \
             for one already uploaded (`vault_blob_put`). A file's name must start with a \
             letter or digit and hold only letters, digits, `.`, `_` and `-`, at most 255. \
             Files per `type`: `gguf_text`: model; `gguf_vision`: model, mmproj; \
             `gguf_vision_speculative`: model, mmproj, optional draft; `mlx`, `torch`, \
             `openvino`, `onnx_genai`: dir_archive; `mlx_speculative`, `torch_speculative`: \
             target_dir_archive, draft_dir_archive; `llamacpp_cli_stock_tools`, `ovms`, \
             `trt_edgellm`: archive, with `flavor`. A packed directory is one tar, tar.gz or \
             zip with the files at its root. The server judges the combination only after the files it \
             lacks are uploaded, so match them to this list first.",
        )
    },
    Route {
        response: Response::Exists,
        ..route(
            "vault",
            "blob-exists",
            HEAD,
            "/api/v1/vault/blobs/{principal_id}/{sha256}",
            "Whether a file is already uploaded for a principal.",
        )
    },
    Route {
        body: Body::Blob(&BLOB_FILE),
        ..route(
            "vault",
            "blob-put",
            PUT,
            "/api/v1/vault/blobs/{principal_id}/{sha256}",
            "Upload one file for a principal, keyed by its digest.\n\
             The digest is computed from the file; a `sha256` given must match it. A file \
             already there is not sent again.",
        )
    },
    Route {
        query: &[OWNER_NAMES, VAULT_OBJECT],
        ..route(
            "vault",
            "show",
            GET,
            "/api/v1/vault/resources/{resource_id}",
            "Read one resource's record: owner, digest and size.",
        )
    },
    Route {
        response: Response::Stream,
        ..route(
            "vault",
            "download",
            GET,
            "/vault/resources/{resource_id}/content",
            "Download a resource's bytes to a local file, checked against the record's digest.\n\
             An existing `out` is refused unless `overwrite` is set. Answers the absolute \
             path, the size and the digest.",
        )
    },
    route(
        "vault",
        "delete",
        DELETE,
        "/api/v1/vault/resources/{resource_id}",
        "Delete a resource's record and every other record of its upload.\n\
         Then deletes each of its files no other record names. Plans naming any of those ids \
         stop resolving, and describing a deleted file again means uploading it again \
         (`vault_blob_put`). An organization's resource needs `manage_orgs` as a member, or \
         `manage_any_orgs`.",
    ),
    // ── plans ───────────────────────────────────────────────────────────────
    Route {
        query: &[
            opt("status", PLAN_STATUS, "Only plans in this status"),
            opt(
                "scope",
                Kind::Enum(PLAN_SCOPES),
                // Values unquoted: the CLI would respell a quoted `all` as `--all`.
                "mine: the plans of your campaigns and your organization's, whoever filed \
                 them, and those you filed outside any campaign. all: every plan, which needs \
                 `read_any_plans`. Left out, the widest your grants allow",
            ),
            opt(
                "submitted_by",
                Kind::String,
                "Only plans this account (`user_…`) filed",
            ),
            opt(
                "campaign_id",
                Kind::String,
                "Only plans filed under this campaign",
            ),
            ORDER,
            SUBMITTER_NAMES,
        ],
        response: Response::Page,
        ..route(
            "plans",
            "list",
            GET,
            "/api/v1/plans",
            "List plans with their status and progress.\nNeeds `read_plans`.",
        )
    },
    Route {
        body: Body::Document(&CHECKED_DOCUMENT),
        ..route(
            "plans",
            "validate",
            POST,
            "/api/v1/plans/validate",
            "Check a plan document as filing checks it, and file nothing.\n\
             Needs `submit_plans`, and no campaign, name or owner. Answers a verdict whatever \
             the outcome: `job_count`, `gated_model_count`, and `issues`, each an `error` or a \
             `warning`, with its line where it has one; the document is valid when no issue is \
             an error. It checks no gated model's token and no campaign or base plan, which the \
             `dry_run` of `campaigns_create` or `campaigns_revise` does. It runs no fleet match \
             and reads no model catalog, so it gives none of a real filing's `warnings` \
             (`fleet`, `uncatalogued_model`).",
        )
    },
    Route {
        response: Response::Text,
        ..route(
            "plans",
            "document",
            GET,
            "/api/v1/plans/{plan_id}/document",
            "Read back the plan document a campaign's plan was filed from.\n\
             Needs `read_plans`, plus `read_clients` or `read_any_clients`, since the document \
             names devices (403 otherwise, for its filer too). A plan filed as job bodies has \
             none (404).",
        )
    },
    Route {
        response: Response::Text,
        ..route(
            "plans",
            "variants",
            GET,
            "/api/v1/plans/{plan_id}/variants",
            "Read a plan's `[[variants]]` blocks as TOML, for another plan document to append.\n\
             Needs what `plans_document` needs, and gives its refusals. A block that names no \
             `benchmarks` carries the plan's top-level list, so the blocks describe the cells \
             the plan ran; a variant edit's `add_from` appends them so. The document's \
             `expires_at`, `merge_duplicates` and comments are left out.",
        )
    },
    Route {
        query: &[SUBMITTER_NAMES],
        ..route(
            "plans",
            "status",
            GET,
            "/api/v1/plans/{plan_id}",
            "Read one plan's status and progress.\nNeeds `read_plans`.",
        )
    },
    Route {
        body: Body::Fields(&[req(
            "note",
            Kind::String,
            "The whole note, at most 4096 characters once trimmed; a blank one clears it",
        )]),
        ..route(
            "plans",
            "update",
            PATCH,
            "/api/v1/plans/{plan_id}",
            "Change a plan's note, the one thing about a plan that changes after filing.\n\
             Needs `submit_plans` and the right to file into the plan's campaign, or to have \
             filed the plan outside one; `manage_any_orgs` reaches any plan. `changed: false` \
             wrote nothing.",
        )
    },
    Route {
        rows: Some("jobs"),
        query: &[
            opt(
                "limit",
                Kind::Integer,
                "Jobs per page (server default 100, at most 500)",
            ),
            opt("offset", Kind::Integer, "Jobs to skip"),
            opt(
                "results",
                Kind::Boolean,
                "Add each job's latest scoring run under `result` (`scored` with its rows, \
                 `not_run` or `unknown`), read from the results warehouse by job id, and what \
                 the plan's page offers beside the jobs: `results_read` (whether that read \
                 answered in time), `rerun_campaign_id`, `may_cancel`, `cancel_permission`, \
                 `may_edit_note` (whether you may change the plan's note), `document_state` and \
                 `names`",
            ),
            opt(
                "all",
                Kind::Boolean,
                "Every job in one page, whatever `limit` and `offset` say",
            ),
        ],
        ..route(
            "plans",
            "jobs",
            GET,
            "/api/v1/plans/{plan_id}/jobs",
            "List a plan's jobs with their state and lease.\n\
             Needs `read_plans`. With `results`, the JSON read of the plan's results.",
        )
    },
    Route {
        response: Response::Text,
        ..route(
            "plans",
            "results",
            GET,
            "/api/v1/plans/{plan_id}/results.csv",
            "Read a plan's results as CSV: a row per metric of each job's latest scoring run.\n\
             Needs `read_plans`; without `read_clients`, what names the device and why a job \
             failed is left empty. Carried-over jobs are included. The rows come from the \
             results warehouse, which refreshes every five minutes by default: a run scored \
             since is missing until the next refresh, and a plan filed since lists nothing. 404 \
             when the server has no warehouse; 502 when the warehouse fails.",
        )
    },
    Route {
        rows: Some("plans"),
        ..route(
            "plans",
            "by-job",
            GET,
            "/api/v1/jobs/{job_id}/plans",
            "List the plans that asked for a job, oldest first, and read the body it ran.\n\
             Needs `read_plans`. Answers for a finished job too. A plan outside your scope is \
             left out, and 404 when none is left. The `body` has every `auth_token` removed.",
        )
    },
    route(
        "plans",
        "explain",
        GET,
        "/api/v1/plans/{plan_id}/explain",
        "Explain why a plan's outstanding jobs are or are not being claimed.\n\
         Needs `read_plans`; `read_clients` adds the fleet detail.",
    ),
    route(
        "plans",
        "cancel",
        POST,
        "/api/v1/plans/{plan_id}/cancel",
        "Cancel a plan.\n\
         Needs `cancel_plans` for a plan of your own campaign, your organization's, or a \
         disabled member's you wind down as your organization's administrator, whoever \
         filed it, or one you filed outside every campaign; `cancel_any_plans` for any \
         other, a deleted campaign's included. 409 if the plan is already complete.",
    ),
    // ── model parameters ────────────────────────────────────────────────────
    Route {
        rows: Some("items"),
        ..route(
            "model-params",
            "list",
            GET,
            "/api/v1/model-params",
            "List every model's parameter counts: the model catalog a scoring pass stamps on \
             results.\n\
             Needs `read_any_plans`. One page, in `{key}.json` byte order (`lfm2-8b` before \
             `lfm2`); `unreadable` counts the rows that did not read. A filing warns \
             `uncatalogued_model` for a model with no row.",
        )
    },
    Route {
        body: Body::Fields(&[
            req(
                "model",
                Kind::String,
                "Any name catalog lookup reads, such as `LiquidAI/LFM3-1.7B-GGUF`, or a vault \
                 resource id (`vault_…`); the row is keyed by its normalized form",
            ),
            TOTAL_MILLIONS,
            opt(
                "active_millions",
                Kind::Integer,
                "Active parameters, in millions, at most the total; the total when left out, as \
                 for a dense model",
            ),
            opt(
                "note",
                Kind::String,
                "Where the counts come from, at most 500 characters; a blank one is none",
            ),
        ]),
        ..route(
            "model-params",
            "create",
            POST,
            "/api/v1/model-params",
            "Add a model's parameter counts.\n\
             Needs `manage_orgs` as a member of the organization owning the vault model the row \
             is keyed by, or `manage_any_orgs`, which every other row takes, a file a person \
             owns included. 409 if the key already has a row: `model_params_update` replaces \
             it.",
        )
    },
    Route {
        body: Body::Fields(&[
            TOTAL_MILLIONS,
            opt(
                "active_millions",
                Kind::Integer,
                "Active parameters, in millions, at most the total; the total when left out",
            ),
            opt(
                "note",
                Kind::String,
                "Where the counts come from, at most 500 characters; left out or blank, the row \
                 keeps none",
            ),
        ]),
        ..route(
            "model-params",
            "update",
            PUT,
            "/api/v1/model-params/{key}",
            "Replace a model's parameter counts and note.\n\
             Needs what `model_params_create` needs. The row is replaced whole: a note left out \
             is dropped, and `active_millions` left out is the total. `changed: false` wrote \
             nothing. 404 if the key has no row.",
        )
    },
    route(
        "model-params",
        "delete",
        DELETE,
        "/api/v1/model-params/{key}",
        "Remove a model's parameter counts.\n\
         Needs what `model_params_create` needs. Answers success whether or not the key had a \
         row. Results scored later carry no counts for the model, and a filing warns \
         `uncatalogued_model`.",
    ),
    // ── the descriptor catalog ──────────────────────────────────────────────
    Route {
        query: &[
            opt(
                "scope",
                Kind::String,
                "Global, or an organization id or slug",
            ),
            opt("kind", Kind::Enums(CATALOG_KINDS), "Entry kinds to keep"),
            opt(
                "state",
                Kind::Enums(CATALOG_STATES),
                "Lifecycle states to keep",
            ),
            opt(
                "descriptor_kind",
                Kind::Enum(DESCRIPTOR_KINDS),
                "Runtime or model entries",
            ),
            opt(
                "type",
                Kind::CommaList,
                "Runtime or model descriptor types to keep",
            ),
            opt("prefix", Kind::String, "Case-insensitive name prefix"),
            opt(
                "members",
                Kind::Boolean,
                "Include full members and plan TOML",
            ),
        ],
        response: Response::Page,
        ..route(
            "catalog",
            "list",
            GET,
            "/api/v1/descriptor-catalog",
            "List descriptor catalog entries, newest release first.",
        )
    },
    Route {
        query: &[req("names", Kind::String, "Bare or qualified catalog name")],
        rows: Some("entries"),
        ..route(
            "catalog",
            "show",
            GET,
            "/api/v1/descriptor-catalog/names",
            "Show one catalog entry and all its members.",
        )
    },
    Route {
        rows: Some("items"),
        ..route(
            "catalog",
            "for-device",
            GET,
            "/api/v1/descriptor-catalog/devices/{client_id}",
            "List catalog members offered to a device, with withheld reasons.\nWithout \
             `client_id`, pipette-ctl reads the registration file of the pipette client in \
             the work directory.",
        )
    },
    // ── the public catalog ──────────────────────────────────────────────────
    Route {
        public: true,
        query: &[opt(
            "type",
            Kind::Enum(BENCHMARK_TYPES),
            "Only benchmarks of this `benchmark_type`",
        )],
        ..route(
            "benchmarks",
            "list",
            GET,
            "/benchmarks",
            "List the benchmark catalog.\n\
             A signed call also lists the caller's organization's private entries.",
        )
    },
    Route {
        public: true,
        ..route(
            "benchmarks",
            "show",
            GET,
            "/benchmarks/{benchmark_id}",
            "Read one benchmark definition.\n\
             An `eval` entry carries its whole sample set under `samples` (often about 1 MB); \
             name `fields` without it to leave it out.",
        )
    },
];
