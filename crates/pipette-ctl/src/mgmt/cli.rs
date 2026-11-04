//! The typed CLI verbs: `pipette-ctl <area> <verb>`, one per catalog row.
//!
//! Built with clap's builder API from [`ROUTES`] rather than derived, so a
//! route's flags are its catalog fields and cannot drift from the MCP schema.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, IsTerminal as _, Write},
    path::Path,
};

use anyhow::Context;
use clap::{Arg, ArgAction, ArgMatches, FromArgMatches, Subcommand, builder::PossibleValuesParser};
use serde_json::{Map, Value};
use tabled::{builder::Builder, settings::Style};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use pipette_mgmt_client::{ImportProgress, ServerUrl};
use pipette_workspace::resolve_work_dir;

use super::{
    Args, Session,
    catalog::{Field, Kind, ROUTES, Response, Route, param_help},
};

/// One area verb, parsed into catalog arguments.
#[derive(Debug)]
pub struct RouteCommand {
    route: &'static Route,
    args: Args,
    server: Option<ServerUrl>,
    json: bool,
    withheld: bool,
}

const SERVER: &str = "server";
const JSON: &str = "json";
const WITHHELD: &str = "withheld";

/// The CLI areas, in catalog order.
fn areas() -> Vec<&'static str> {
    let mut areas: Vec<&'static str> = Vec::new();
    for route in ROUTES.iter().filter(|route| route.has_verb()) {
        if !areas.contains(&route.area) {
            areas.push(route.area);
        }
    }
    areas
}

fn area_about(area: &str) -> &'static str {
    match area {
        "clients" => "Read and administer the device fleet on the management server",
        "users" => "Read and administer accounts, their keys and pre-auth keys",
        "orgs" => {
            "Manage organizations, their members, devices, private benchmarks and device \
             pre-auth keys, and read their audit logs"
        }
        "audit" => "Read the whole audit log: who changed what, refused requests included",
        "campaigns" => "Manage campaigns: stable ids that point at the plan being executed",
        "vault" => "Register, list, read and delete private artifacts in the vault",
        "plans" => {
            "Check, read, explain and cancel plans, read their results, and change their notes"
        }
        "model-params" => "Read and edit each model's parameter counts, the model catalog",
        "benchmarks" => "Read the benchmark catalog",
        "catalog" => "Read runtime and model descriptor catalog entries",
        _ => "Management server routes",
    }
}

impl FromArgMatches for RouteCommand {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        let missing = || clap::Error::new(clap::error::ErrorKind::MissingSubcommand);
        let (area, matches) = matches.subcommand().ok_or_else(missing)?;
        let (verb, matches) = matches.subcommand().ok_or_else(missing)?;
        let route = ROUTES
            .iter()
            .find(|route| route.has_verb() && route.area == area && route.verb == verb)
            .ok_or_else(|| clap::Error::new(clap::error::ErrorKind::InvalidSubcommand))?;
        let args = collect(route, matches).map_err(|error| {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                format!("{error:#}\n"),
            )
        })?;
        Ok(Self {
            route,
            args,
            server: matches.get_one::<ServerUrl>(SERVER).cloned(),
            json: matches.get_flag(JSON),
            withheld: route.area == "catalog"
                && route.verb == "for-device"
                && matches.get_flag(WITHHELD),
        })
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

impl Subcommand for RouteCommand {
    fn augment_subcommands(cmd: clap::Command) -> clap::Command {
        areas().into_iter().fold(cmd, |cmd, area| {
            let verbs = ROUTES
                .iter()
                .filter(|route| route.has_verb() && route.area == area)
                .map(route_command);
            cmd.subcommand(
                clap::Command::new(area)
                    .about(area_about(area))
                    .subcommand_required(true)
                    .arg_required_else_help(true)
                    .subcommands(verbs),
            )
        })
    }

    fn augment_subcommands_for_update(cmd: clap::Command) -> clap::Command {
        Self::augment_subcommands(cmd)
    }

    fn has_subcommand(name: &str) -> bool {
        ROUTES
            .iter()
            .any(|route| route.has_verb() && route.area == name)
    }
}

fn route_command(route: &'static Route) -> clap::Command {
    let mut cmd = clap::Command::new(route.verb).after_help(format!(
        "{} {}  (MCP tool `{}`)",
        route.method.as_str(),
        route.path,
        route.tool()
    ));
    for name in route.path_params() {
        cmd = cmd.arg(
            Arg::new(name)
                .required(!route.derives(name))
                .value_name(name.to_uppercase())
                .help(param_help(name)),
        );
    }
    for field in route.fields() {
        if route.area == "catalog" && route.verb == "show" && field.name == "names" {
            cmd = cmd.arg(
                Arg::new("names")
                    .required(true)
                    .value_name("NAME")
                    .help(field.help),
            );
        } else {
            cmd = cmd.arg(field_arg(field));
        }
    }
    let json = if route.response == Response::Page {
        "Print the server's answer as JSON (under `--all`, every page merged into one `items` \
         envelope)"
    } else {
        "Print the server's answer as JSON"
    };
    let cmd = cmd
        .arg(
            Arg::new(SERVER)
                .long(SERVER)
                .env("PIPETTE_MGMT_SERVER")
                .value_name("URL")
                .value_parser(|raw: &str| ServerUrl::try_new(raw).map_err(|e| e.to_string()))
                .help("Management server URL (default: the one this working directory is registered against)"),
        )
        .arg(
            Arg::new(JSON)
                .long(JSON)
                .action(ArgAction::SetTrue)
                .help(json),
        );
    let cmd = if route.area == "catalog" && route.verb == "for-device" {
        cmd.arg(
            Arg::new(WITHHELD)
                .long(WITHHELD)
                .action(ArgAction::SetTrue)
                .help("Print every withheld member and its reason"),
        )
    } else {
        cmd
    };
    let spellings = spellings(&cmd);
    cmd.about(respell(&spellings, route.summary()))
        .long_about(respell(&spellings, route.about))
        .mut_args(|arg| match arg.get_help().map(ToString::to_string) {
            Some(help) => arg.help(respell(&spellings, &help)),
            None => arg,
        })
}

/// Each name the catalog's text may put in backticks, as the CLI spells it:
/// a tool as its command, an argument of `cmd` as its flag or positional.
fn spellings(cmd: &clap::Command) -> Vec<(String, String)> {
    let tools = ROUTES.iter().map(|route| (route.tool(), route.command()));
    let args = cmd.get_arguments().map(|arg| {
        let id = arg.get_id().as_str();
        let spelled = arg
            .get_long()
            .map_or_else(|| id.to_uppercase(), |long| format!("--{long}"));
        (id.to_string(), spelled)
    });
    tools.chain(args).collect()
}

/// `text` with each backticked name in `spellings` spelled the CLI's way.
fn respell(spellings: &[(String, String)], text: &str) -> String {
    text.split('`')
        .enumerate()
        .map(|(index, part)| {
            let quoted = index % 2 == 1;
            spellings
                .iter()
                .find(|(name, _)| quoted && name == part)
                .map_or(part, |(_, spelled)| spelled.as_str())
        })
        .collect::<Vec<_>>()
        .join("`")
}

/// Bare means true; a value follows `=` (`--dry-run=false`), so a positional
/// after the flag is never read as its value.
fn switch(arg: Arg) -> Arg {
    arg.value_name("true|false")
        .num_args(0..=1)
        .require_equals(true)
        .default_missing_value("true")
        .value_parser(clap::value_parser!(bool))
}

fn field_arg(field: &Field) -> Arg {
    // A list flag given no times is the empty list, so a required one can
    // still be cleared from the command line.
    let list = matches!(field.kind, Kind::Strings | Kind::Enums(_));
    let arg = Arg::new(field.name)
        .long(field.name.replace('_', "-"))
        .required(field.required && !list)
        .help(field.help);
    match field.kind {
        Kind::String => arg.value_name("TEXT"),
        Kind::Clearable => arg.value_name("TEXT|none"),
        Kind::Integer => arg.value_name("N").value_parser(clap::value_parser!(u64)),
        Kind::Boolean => switch(arg),
        Kind::Enum(allowed) => arg
            .value_name(field.name.to_uppercase())
            .value_parser(PossibleValuesParser::new(allowed)),
        Kind::Strings => arg.value_name("TEXT").action(ArgAction::Append),
        Kind::CommaList => arg
            .value_name("ID")
            .value_delimiter(',')
            .action(ArgAction::Append),
        Kind::Enums(allowed) => arg
            .value_name(field.name.to_uppercase())
            .value_delimiter(',')
            .action(ArgAction::Append)
            .value_parser(PossibleValuesParser::new(allowed)),
        Kind::Keys => arg
            .value_name("KEY")
            .value_delimiter(',')
            .action(ArgAction::Append),
        Kind::Pairs => arg.value_name("KEY=VALUE|JSON").action(ArgAction::Append),
        Kind::JsonOrToml => arg.value_name("JSON|TOML|PATH"),
        Kind::Owner { .. } => arg.value_name("OWNER"),
        Kind::Path => arg.value_name("PATH"),
        Kind::Toml => arg.value_name("TOML|PATH"),
        Kind::VaultFile => arg.value_name("PATH|JSON"),
    }
}

/// The catalog arguments clap parsed, typed as the MCP front end would send
/// them.
fn collect(route: &Route, matches: &ArgMatches) -> anyhow::Result<Args> {
    let mut args = Map::new();
    for name in route.path_params() {
        if let Some(value) = matches.get_one::<String>(name) {
            args.insert(name.to_string(), Value::String(value.clone()));
        }
    }
    for field in route.fields() {
        let name = field.name;
        let value = match field.kind {
            Kind::String
            | Kind::Clearable
            | Kind::Enum(_)
            | Kind::Owner { .. }
            | Kind::Path
            | Kind::Toml => matches
                .get_one::<String>(name)
                .map(|value| Value::String(value.clone())),
            // A file already uploaded is named by its `{filename, sha256}`.
            Kind::VaultFile => matches
                .get_one::<String>(name)
                .map(|raw| {
                    if raw.trim_start().starts_with('{') {
                        json_arg(name, raw)
                    } else {
                        Ok(Value::String(raw.clone()))
                    }
                })
                .transpose()?,
            Kind::Integer => matches
                .get_one::<u64>(name)
                .map(|value| Value::from(*value)),
            Kind::Boolean => matches
                .get_one::<bool>(name)
                .map(|value| Value::Bool(*value)),
            Kind::Strings | Kind::CommaList | Kind::Enums(_) | Kind::Keys => matches
                .get_many::<String>(name)
                .map(|values| values.cloned().map(Value::String).collect())
                .or_else(|| field.required.then(|| Value::Array(Vec::new()))),
            Kind::Pairs => matches
                .get_many::<String>(name)
                .map(|raws| pairs_arg(name, raws))
                .transpose()?,
            // `@` is optional: the invoker reads the file.
            Kind::JsonOrToml => matches
                .get_one::<String>(name)
                .map(|raw| Value::String(raw.strip_prefix('@').unwrap_or(raw).to_string())),
        };
        if let Some(value) = value {
            args.insert(name.to_string(), value);
        }
    }
    Ok(args)
}

/// Repeated `key=value` flags, or JSON objects, as the one object they spell.
fn pairs_arg<'a>(name: &str, mut raws: impl Iterator<Item = &'a String>) -> anyhow::Result<Value> {
    raws.try_fold(Map::new(), |mut pairs, raw| {
        pair_object(name, raw)?
            .into_iter()
            .try_for_each(|(key, value)| {
                anyhow::ensure!(
                    pairs.insert(key.clone(), value).is_none(),
                    "--{name} names `{key}` twice"
                );
                Ok(())
            })?;
        Ok(pairs)
    })
    .map(Value::Object)
}

/// One flag value: a JSON object, or one `key=value` pair.
fn pair_object(name: &str, raw: &str) -> anyhow::Result<Map<String, Value>> {
    if raw.trim_start().starts_with('{') {
        return match json_arg(name, raw)? {
            Value::Object(object) => Ok(object),
            other @ (Value::Null
            | Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Array(_)) => anyhow::bail!("--{name} must be a JSON object, got {other}"),
        };
    }
    // A value may hold `=`; a key that does is spelled as JSON.
    let (key, value) = raw
        .split_once('=')
        .with_context(|| format!("--{name} takes KEY=VALUE or a JSON object, got `{raw}`"))?;
    Ok(Map::from_iter([(
        key.to_string(),
        Value::String(value.to_string()),
    )]))
}

/// A JSON flag's value: inline, or read from the file after `@`.
fn json_arg(name: &str, raw: &str) -> anyhow::Result<Value> {
    let text = match raw.strip_prefix('@') {
        Some(path) => {
            fs::read_to_string(path).with_context(|| format!("to read --{name} {path}"))?
        }
        None => raw.to_string(),
    };
    serde_json::from_str(&text).with_context(|| format!("--{name} is not valid JSON"))
}

impl RouteCommand {
    pub fn execute(self, work_dir: Option<&Path>) -> anyhow::Result<()> {
        let session = Session::open(&resolve_work_dir(work_dir)?, self.server)?;
        let stderr = io::stderr();
        // A Windows console can print the escape that clears a line as text.
        let columns =
            (cfg!(unix) && stderr.is_terminal()).then_some(terminal_columns as fn() -> usize);
        let mut progress = ProgressLine::new(stderr, columns);
        let answer =
            session.invoke_reporting(self.route, &self.args, &mut |event| progress.report(event));
        progress.finish();
        let answer = answer?;
        answer.notes.iter().for_each(|note| eprintln!("{note}"));
        if self.json {
            println!("{}", serde_json::to_string_pretty(&answer.value)?);
            return Ok(());
        }
        if self.route.area == "catalog" {
            print!(
                "{}",
                catalog_output(self.route, &answer.value, self.withheld)
            );
        } else {
            print_answer(self.route, &answer.value);
        }
        // A script reads `unknown` from the JSON; a person gets an error.
        if self.route.area == "catalog"
            && self.route.verb == "show"
            && let Some(unknown) = unknown_catalog_names(&answer.value)
        {
            anyhow::bail!("the server has no catalog entry named {unknown}");
        }
        Ok(())
    }
}

/// The progress of an import. A header line names the source once, since it
/// can be long, and each report names only the file. On a Unix terminal a
/// report rewrites its file's line, cut by `fit` to the terminal width since
/// the rewrite cannot reach a wrapped line; elsewhere each report is a line.
struct ProgressLine<W: Write> {
    out: W,
    /// The width of the terminal, read at each report, or `None` for a pipe
    /// or a file.
    columns: Option<fn() -> usize>,
    /// The `hf_ref` the last header line names.
    header: Option<String>,
    /// The `hf_ref` and the file of the terminal line that has no newline yet.
    open: Option<(String, String)>,
}

impl<W: Write> ProgressLine<W> {
    fn new(out: W, columns: Option<fn() -> usize>) -> Self {
        Self {
            out,
            columns,
            header: None,
            open: None,
        }
    }

    /// Write `event`. A write that fails does not fail the call: the line
    /// only tells the user how far the import is.
    fn report(&mut self, event: &ImportProgress) {
        let _ = self.write(event);
    }

    fn write(&mut self, event: &ImportProgress) -> io::Result<()> {
        let file = (event.hf_ref.clone(), event.file.clone());
        if self.open.as_ref().is_some_and(|open| *open != file) {
            self.open = None;
            writeln!(self.out)?;
        }
        if self.header.as_ref() != Some(&event.hf_ref) {
            writeln!(self.out, "importing {}", event.hf_ref)?;
            self.header = Some(event.hf_ref.clone());
        }
        let percent = event
            .bytes_done
            .saturating_mul(100)
            .checked_div(event.bytes_total)
            .unwrap_or(100);
        let counts = format!(
            ": {} of {} bytes ({percent}%)",
            event.bytes_done, event.bytes_total
        );
        let Some(columns) = self.columns else {
            return writeln!(self.out, "  {}{counts}", event.file);
        };
        // The last column stays empty: a terminal can wrap a line that fills it.
        let line = fit(&event.file, &counts, columns().saturating_sub(1));
        // `\x1b[K` clears what a longer earlier report left on the line.
        write!(self.out, "\r{line}\x1b[K")?;
        if event.bytes_done >= event.bytes_total {
            writeln!(self.out)?;
            self.open = None;
        } else {
            self.open = Some(file);
        }
        self.out.flush()
    }

    /// End the open terminal line. The next output then starts on a new line.
    fn finish(&mut self) {
        if self.open.take().is_some() {
            let _ = writeln!(self.out);
        }
    }
}

/// The progress line of `file`, at most `width` terminal columns. A long name
/// keeps its end, which tells files apart; past that the line's end is cut.
fn fit(file: &str, counts: &str, width: usize) -> String {
    let line = format!("  {file}{counts}");
    if line.width() <= width {
        return line;
    }
    let room = width.saturating_sub(counts.width() + 3);
    if room == 0 {
        let mut used = 0;
        return line
            .chars()
            .take_while(|c| {
                used += c.width().unwrap_or(0);
                used <= width
            })
            .collect();
    }
    let mut used = 0;
    let start = file
        .char_indices()
        .rev()
        .take_while(|(_, c)| {
            used += c.width().unwrap_or(0);
            used <= room
        })
        .last()
        .map_or(file.len(), |(start, _)| start);
    let tail = file.get(start..).unwrap_or_default();
    format!("  \u{2026}{tail}{counts}")
}

/// The width of the terminal on stderr, in columns, or 80 if unknown.
fn terminal_columns() -> usize {
    #[cfg(unix)]
    let columns = rustix::termios::tcgetwinsize(io::stderr())
        .ok()
        .map(|size| usize::from(size.ws_col));
    #[cfg(not(unix))]
    let columns = None;
    columns.filter(|columns| *columns > 0).unwrap_or(80)
}

/// An answer for a person: a listing as a table, a record as `key: value`
/// lines, a text answer (a plan's document) as itself. The shape follows the
/// route, not the data, so an empty list prints like a full one.
fn print_answer(route: &Route, value: &Value) {
    match value {
        Value::Null => println!("done"),
        Value::String(text) => print!("{text}"),
        Value::Array(rows) => print_rows(rows),
        Value::Object(object) => {
            let rows = route
                .rows_key()
                .and_then(|key| Some((key, object.get(key)?.as_array()?)));
            match rows {
                Some((key, rows)) => print_listing(object, key, rows),
                None => print_record(object),
            }
        }
        Value::Bool(_) | Value::Number(_) => println!("{value}"),
    }
}

fn catalog_output(route: &Route, value: &Value, show_withheld: bool) -> String {
    let mut out = String::new();
    let for_device = route.verb == "for-device";
    let entries = route
        .rows_key()
        .and_then(|key| value.get(key))
        .and_then(Value::as_array);
    if let Some(entries) = entries {
        if entries.is_empty() {
            out.push_str(if for_device {
                "(no catalog entries offered to this device)\n"
            } else {
                "(no entries)\n"
            });
        }
        entries.iter().for_each(|entry| {
            let label = entry
                .get("display_name")
                .or_else(|| entry.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("(unnamed)");
            out.push_str(label);
            [
                "scope",
                "descriptor_kind",
                "kind",
                "state",
                "release_time",
                "created_at",
                "created_by",
                "member_count",
            ]
            .iter()
            .filter_map(|key| Some((key, entry.get(*key)?)))
            .for_each(|(key, value)| out.push_str(&format!("  {key}: {}", render(value))));
            out.push('\n');
            let (members, full) = match entry.get("members").and_then(Value::as_array) {
                Some(members) => (Some(members), true),
                None => (
                    entry.get("member_summaries").and_then(Value::as_array),
                    false,
                ),
            };
            members
                .into_iter()
                .flatten()
                .for_each(|member| catalog_member_line(&mut out, member, full));
            out.push('\n');
        });
    }
    if for_device && let Some(withheld) = value.get("withheld") {
        out.push_str(&format!(
            "withheld: {} members in {} entries\n",
            withheld.get("member_count").map(render).unwrap_or_default(),
            withheld.get("entry_count").map(render).unwrap_or_default()
        ));
        let members = withheld
            .get("members")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let reasons = members
            .iter()
            .filter_map(|member| member.get("reason_code").and_then(Value::as_str))
            .fold(BTreeMap::<&str, usize>::new(), |mut counts, code| {
                *counts.entry(code).or_default() += 1;
                counts
            });
        if !reasons.is_empty() {
            let summary = reasons
                .into_iter()
                .map(|(code, count)| format!("{code} ({count})"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("reasons: {summary}\n"));
        }
        if show_withheld {
            out.extend(members.iter().map(|member| {
                format!(
                    "  {}: {} ({})\n",
                    member
                        .get("selector")
                        .and_then(Value::as_str)
                        .unwrap_or("(unnamed)"),
                    member.get("reason").map(render).unwrap_or_default(),
                    member.get("reason_code").map(render).unwrap_or_default()
                )
            }));
        } else if withheld
            .get("member_count")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
        {
            out.push_str("pass --withheld to show individual reasons\n");
        }
    }
    let listing = value.as_object();
    out.extend(listing.and_then(unreadable_line).map(|line| line + "\n"));
    out.extend(listing.and_then(more_rows_line).map(|line| line + "\n"));
    out
}

/// The names a `catalog show` answer did not find, each in backticks.
fn unknown_catalog_names(value: &Value) -> Option<String> {
    let names = value.get("unknown")?.as_array()?;
    (!names.is_empty()).then(|| {
        names
            .iter()
            .map(|name| format!("`{}`", render(name)))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn catalog_member_line(out: &mut String, member: &Value, full: bool) {
    out.push_str("  ");
    out.push_str(
        member
            .get("selector")
            .and_then(Value::as_str)
            .unwrap_or("(unnamed member)"),
    );
    if let Some(coverage) = member.get("coverage") {
        out.push_str(&format!("  coverage: {}", render(coverage)));
    }
    out.push('\n');
    if full {
        [
            "member_key",
            "digest",
            "added_at",
            "descriptor",
            "model_key",
            "parameters",
            "offered_device_count",
        ]
        .iter()
        .filter_map(|key| Some((key, member.get(*key)?)))
        .for_each(|(key, value)| out.push_str(&format!("    {key}: {}\n", render(value))));
        if let Some(toml) = member.get("plan_toml").and_then(Value::as_str) {
            out.push_str("    plan TOML: ");
            out.push_str(toml);
            out.push('\n');
        }
        if let Some(evidence) = member.get("evidence") {
            out.push_str(&format!("    evidence: {}\n", render(evidence)));
        }
    }
}

/// A listing: its other fields, the rows as a table, then how to page on.
fn print_listing(object: &Map<String, Value>, key: &str, rows: &[Value]) {
    const PAGE_KEYS: &[&str] = &["next_cursor", "unreadable", "unreadable_benchmark_ids"];
    object
        .iter()
        .filter(|(name, value)| {
            name.as_str() != key && !PAGE_KEYS.contains(&name.as_str()) && !value.is_null()
        })
        .for_each(|(name, value)| println!("{name}: {}", render(value)));
    print_rows(rows);
    if let Some(line) = unreadable_line(object) {
        println!("{line}");
    }
    if let Some(line) = more_rows_line(object) {
        println!("{line}");
    }
}

/// How to read on, when the server holds another page.
fn more_rows_line(listing: &Map<String, Value>) -> Option<String> {
    let cursor = listing.get("next_cursor")?.as_str()?;
    Some(format!("more rows: pass --cursor {cursor}, or --all"))
}

/// A page counts the records it could not decode; an organization's catalog
/// names the entries it keeps but cannot read.
fn unreadable_line(listing: &Map<String, Value>) -> Option<String> {
    let count = listing
        .get("unreadable")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0);
    let ids = listing
        .get("unreadable_benchmark_ids")
        .filter(|ids| ids.as_array().is_some_and(|ids| !ids.is_empty()));
    count
        .map(|n| format!("unreadable: {n} record(s) the server could not decode"))
        .or_else(|| {
            ids.map(|ids| {
                format!(
                    "unreadable: {} (kept, but the server cannot read them)",
                    render(ids)
                )
            })
        })
}

fn print_record(object: &Map<String, Value>) {
    print!("{}", record(object));
}

/// A list of texts that span lines (a campaign's `[[variants]]` blocks)
/// prints each under `# key[index]`, the index `remove` takes, since a `,`
/// join runs them together.
fn record(object: &Map<String, Value>) -> String {
    object
        .iter()
        .flat_map(|(name, value)| pieces(name, value))
        // A blank line on either side of a block.
        .scan(None, |previous: &mut Option<bool>, piece| {
            let gap = previous
                .replace(piece.block)
                .is_some_and(|before| before || piece.block);
            Some(if gap {
                format!("\n{}", piece.text)
            } else {
                piece.text
            })
        })
        .collect()
}

/// A `key: value` line of a record, or one block of a multi-line text.
struct Piece {
    block: bool,
    text: String,
}

fn pieces(name: &str, value: &Value) -> Vec<Piece> {
    match text_blocks(value) {
        Some(blocks) => blocks
            .iter()
            .enumerate()
            .map(|(index, block)| Piece {
                block: true,
                text: format!("# {name}[{index}]\n{}\n", block.trim_end_matches('\n')),
            })
            .collect(),
        None => {
            let text = if value.as_array().is_some_and(Vec::is_empty) {
                "(none)".to_string()
            } else {
                render(value)
            };
            vec![Piece {
                block: false,
                text: format!("{name}: {text}\n"),
            }]
        }
    }
}

fn text_blocks(value: &Value) -> Option<Vec<&str>> {
    let texts = value
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?;
    texts
        .iter()
        .any(|text| text.contains('\n'))
        .then_some(texts)
}

/// Records as a table, one column per key any row has, in first-seen order.
fn print_rows(rows: &[Value]) {
    if rows.is_empty() {
        println!("(no rows)");
        return;
    }
    if !rows.iter().all(Value::is_object) {
        rows.iter().for_each(|row| println!("{}", render(row)));
        return;
    }
    let mut columns: Vec<&str> = Vec::new();
    for key in rows.iter().filter_map(Value::as_object).flat_map(Map::keys) {
        if !columns.contains(&key.as_str()) {
            columns.push(key);
        }
    }
    let mut table = Builder::default();
    table.push_record(columns.iter().map(|column| column.to_uppercase()));
    for row in rows {
        table.push_record(columns.iter().map(|column| {
            row.get(*column)
                .map(|value| cell(column, value))
                .unwrap_or_default()
        }));
    }
    println!("{}", table.build().with(Style::psql()));
}

/// Longest cell a table shows whole, so a key or a capability list does not
/// set the width of the table. `--json` is never cut.
const CELL_MAX_CHARS: usize = 48;
/// An object's pairs share a wider budget, which a plan's progress counts fit.
const OBJECT_CELL_MAX_CHARS: usize = 160;

/// One table cell, on one line. An id or digest stays whole, since the next
/// command takes it; anything else is cut. An owner shows as its id; any other
/// object as its `key=value` leaves, each cut, the whole cut at
/// [`OBJECT_CELL_MAX_CHARS`], and its own ids and digests left to `--json`.
fn cell(key: &str, value: &Value) -> String {
    if let Value::Object(object) = value {
        if let Some(id) = owner_id(object) {
            return id.to_string();
        }
        let mut leaves = Vec::new();
        push_leaves("", object, &mut leaves);
        return cut(&leaves.join(","), OBJECT_CELL_MAX_CHARS);
    }
    let text = render(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if is_id_or_digest(key) {
        return text;
    }
    cut(&text, CELL_MAX_CHARS)
}

fn is_id_or_digest(key: &str) -> bool {
    key.ends_with("_id") || key.ends_with("sha256") || key == "digest"
}

fn cut(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// `object`'s values as `key=value`, a nested object's under dotted keys. A
/// `type` comes first: it says what the rest is, so the cut never takes it.
fn push_leaves(prefix: &str, object: &Map<String, Value>, leaves: &mut Vec<String>) {
    let (tag, rest): (Vec<_>, Vec<_>) = object.iter().partition(|(key, _)| *key == "type");
    for (key, value) in tag.into_iter().chain(rest) {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Null => {}
            Value::Array(items) if items.is_empty() => {}
            Value::Object(inner) if owner_id(inner).is_none() => {
                push_leaves(&path, inner, leaves);
            }
            _ if is_id_or_digest(key) => {}
            Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Array(_)
            | Value::Object(_) => leaves.push(format!("{path}={}", cell(key, value))),
        }
    }
}

/// The id an owner `{type, user_id | org_id}` names; its prefix says which.
fn owner_id(object: &Map<String, Value>) -> Option<&str> {
    let kind = object.get("type")?.as_str()?;
    let id = object.get(&format!("{kind}_id"))?.as_str()?;
    (object.len() == 2 && matches!(kind, "user" | "org")).then_some(id)
}

/// A value as text: a list joined with `,`, an object of scalars as
/// `key=value` pairs, anything deeper as compact JSON.
fn render(value: &Value) -> String {
    let is_scalar = |value: &Value| !matches!(value, Value::Array(_) | Value::Object(_));
    match value {
        Value::Array(items) if items.iter().all(is_scalar) => {
            items.iter().map(scalar).collect::<Vec<_>>().join(",")
        }
        Value::Object(object) if object.values().all(is_scalar) => object
            .iter()
            .map(|(key, value)| format!("{key}={}", scalar(value)))
            .collect::<Vec<_>>()
            .join(","),
        Value::Array(_) | Value::Object(_) => value.to_string(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => scalar(value),
    }
}

fn scalar(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

/// `pipette-ctl mcp`: the same routes, as MCP tools.
#[derive(clap::Args, Debug)]
pub struct McpArgs {
    #[command(subcommand)]
    pub command: McpCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum McpCommand {
    /// Serve the management routes as MCP tools on stdio.
    ///
    /// Signs as the account in this working directory, so register the server
    /// with it named, e.g.
    /// `claude mcp add pipette -- pipette-ctl --work-dir ~/pipette mcp serve`,
    /// or through `PIPETTE_WORK_DIR` and `PIPETTE_MGMT_SERVER` for a client
    /// that drops or takes over the flags.
    Serve {
        /// Management server URL (default: the one this working directory is
        /// registered against).
        #[arg(long, env = "PIPETTE_MGMT_SERVER")]
        server: Option<ServerUrl>,
    },
    /// List the tools `serve` offers.
    Tools {
        /// Print the full MCP tool definitions, input schemas included.
        #[arg(long)]
        json: bool,
    },
}

impl McpArgs {
    pub fn execute(self, work_dir: Option<&Path>) -> anyhow::Result<()> {
        match self.command {
            McpCommand::Serve { server } => {
                let session = Session::open(&resolve_work_dir(work_dir)?, server)?;
                super::mcp::serve(&session, std::io::stdin().lock(), std::io::stdout())
            }
            McpCommand::Tools { json: true } => {
                let tools = ROUTES.iter().map(super::mcp::tool).collect::<Vec<_>>();
                println!("{}", serde_json::to_string_pretty(&tools)?);
                Ok(())
            }
            McpCommand::Tools { json: false } => {
                let mut table = Builder::default();
                table.push_record(["TOOL", "ROUTE", "SUMMARY"]);
                for route in ROUTES {
                    table.push_record([
                        route.tool(),
                        format!("{} {}", route.method.as_str(), route.path),
                        route.summary().to_string(),
                    ]);
                }
                println!("{}", table.build().with(Style::psql()));
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::cli::{Cli, Command};

    fn parse(argv: &[&str]) -> anyhow::Result<RouteCommand> {
        let cli = Cli::try_parse_from(std::iter::once("pipette-ctl").chain(argv.iter().copied()))?;
        let Command::Mgmt(command) = cli.command else {
            anyhow::bail!("parsed as {:?}", cli.command);
        };
        Ok(command)
    }

    #[test]
    fn catalog_commands_parse_as_shared_routes() -> anyhow::Result<()> {
        let list = parse(&[
            "catalog",
            "list",
            "--kind",
            "stable,daily",
            "--descriptor-kind",
            "model",
            "--members",
            "--all",
            "--json",
        ])?;
        assert_eq!(list.route.tool(), "catalog_list");
        assert_eq!(
            Value::Object(list.args),
            json!({"kind": ["stable", "daily"], "descriptor_kind": "model", "members": true, "all": true})
        );
        assert!(list.json);
        let show = parse(&["catalog", "show", "global/Example"])?;
        assert_eq!(Value::Object(show.args), json!({"names": "global/Example"}));
        let device = parse(&["catalog", "for-device"])?;
        assert!(!device.args.contains_key("client_id"));
        assert!(!device.withheld);
        let device = parse(&["catalog", "for-device", "--withheld"])?;
        assert!(device.withheld);
        assert!(!device.args.contains_key("withheld"));
        Ok(())
    }

    const WITHHELD_ANSWER: &str = r#"{"items": [], "withheld": {"member_count": 1, "entry_count": 1, "members": [{"selector": "x::linux-rocm", "reason_code": "no_rule", "reason": "no rule for linux-rocm"}]}, "next_cursor": null, "unreadable": 0}"#;

    #[rstest]
    #[case::list_summary(
        "catalog_list",
        r#"{"items": [{"display_name": "global/x", "scope": "global", "descriptor_kind": "runtime", "kind": "stable", "state": "active", "member_count": 1, "member_summaries": [{"selector": "global/x::macos-arm64"}]}], "next_cursor": "c1", "unreadable": 0}"#,
        false,
        &["global/x  scope: global", "global/x::macos-arm64", "more rows: pass --cursor c1"],
        &["plan TOML"],
    )]
    #[case::show_full_members(
        "catalog_show",
        r#"{"entries": [{"display_name": "x", "scope": "org_1", "members": [{"selector": "x::gguf::q4_0", "digest": "e3b0", "added_at": "2026-11-01T00:00:00Z", "descriptor": {"type": "gguf"}, "evidence": {"runs": 2}, "plan_toml": "{ type = \"gguf\", name = \"x\" }"}]}], "unknown": []}"#,
        false,
        &[
            "plan TOML: { type = \"gguf\", name = \"x\" }",
            "digest: e3b0\n",
            "added_at: 2026-11-01T00:00:00Z\n",
            "descriptor: type=gguf\n",
            "evidence: runs=2\n",
        ],
        &["\"e3b0\""],
    )]
    #[case::device_offered(
        "catalog_for_device",
        r#"{"items": [{"display_name": "global/x", "members": [{"selector": "global/x::macos-arm64", "plan_toml": "{ type = \"llamacpp\" }"}]}], "withheld": {"member_count": 0, "entry_count": 0, "members": []}, "next_cursor": null, "unreadable": 0}"#,
        false,
        &["global/x\n", "plan TOML: { type = \"llamacpp\" }", "withheld: 0 members in 0 entries"],
        &["pass --withheld", "no catalog entries"],
    )]
    #[case::device_withheld_summary(
        "catalog_for_device",
        WITHHELD_ANSWER,
        false,
        &["withheld: 1 members in 1 entries", "reasons: no_rule (1)", "pass --withheld"],
        &["no rule for linux-rocm"],
    )]
    #[case::device_withheld_detail(
        "catalog_for_device",
        WITHHELD_ANSWER,
        true,
        &["no rule for linux-rocm (no_rule)"],
        &["pass --withheld"],
    )]
    fn catalog_human_output(
        #[case] tool: &str,
        #[case] answer: &str,
        #[case] show_withheld: bool,
        #[case] contains: &[&str],
        #[case] lacks: &[&str],
    ) -> anyhow::Result<()> {
        let route = ROUTES
            .iter()
            .find(|route| route.tool() == tool)
            .with_context(|| format!("no route {tool}"))?;
        let output = catalog_output(route, &serde_json::from_str(answer)?, show_withheld);
        contains
            .iter()
            .for_each(|text| assert!(output.contains(text), "{output:?} lacks {text:?}"));
        lacks
            .iter()
            .for_each(|text| assert!(!output.contains(text), "{output:?} has {text:?}"));
        Ok(())
    }

    #[rstest]
    #[case::all_found(json!({"entries": [{"name": "a"}], "unknown": []}), None)]
    #[case::one_unknown(
        json!({"entries": [{"name": "a"}], "unknown": ["b"]}),
        Some("`b`"),
    )]
    #[case::two_unknown(json!({"entries": [], "unknown": ["b", "c"]}), Some("`b`, `c`"))]
    fn show_names_only_the_unknown_names(#[case] answer: Value, #[case] expected: Option<&str>) {
        assert_eq!(unknown_catalog_names(&answer).as_deref(), expected);
    }

    #[test]
    fn flags_become_the_arguments_mcp_would_send() -> anyhow::Result<()> {
        let command = parse(&[
            "clients", "list", "--tag", "lab", "--tag", "rack-2", "--limit", "5", "--all",
        ])?;
        assert_eq!(command.route.tool(), "clients_list");
        assert_eq!(
            Value::Object(command.args),
            json!({ "tag": ["lab", "rack-2"], "limit": 5, "all": true })
        );

        let command = parse(&["orgs", "update", "acme", "--status", "disabled", "--json"])?;
        assert!(command.json);
        assert_eq!(
            Value::Object(command.args),
            json!({ "org_id": "acme", "status": "disabled" })
        );
        Ok(())
    }

    /// This is how a permission set is cleared.
    #[rstest]
    #[case::user(
        &["users", "set-permissions", "user_1"],
        json!({ "user_id": "user_1", "permissions": [] }),
    )]
    #[case::device(
        &["clients", "set-permissions", "ev1_a"],
        json!({ "client_id": "ev1_a", "permissions": [] }),
    )]
    fn a_required_list_left_out_is_sent_empty(
        #[case] argv: &[&str],
        #[case] expected: Value,
    ) -> anyhow::Result<()> {
        assert_eq!(Value::Object(parse(argv)?.args), expected);
        Ok(())
    }

    /// mgmt refuses a membership change that names no organization, so
    /// clearing one is spelled `none` rather than left out.
    #[test]
    fn set_org_requires_an_org_or_none() -> anyhow::Result<()> {
        let command = parse(&["users", "set-org", "user_1", "--org-id", "none"])?;
        assert_eq!(
            Value::Object(command.args),
            json!({ "user_id": "user_1", "org_id": "none" })
        );
        assert!(parse(&["users", "set-org", "user_1"]).is_err());
        Ok(())
    }

    #[rstest]
    #[case::revision_plans(
        &[
            "campaigns",
            "variants",
            "campaign_1",
            "--base-plan-id",
            "plan-0",
            "--carry-earlier",
            "--inherit-from",
            "plan-1,plan-2",
            "--add-from",
            "plan-3",
            "--add-from",
            "plan-4",
        ],
        json!({
            "campaign_id": "campaign_1",
            "base_plan_id": "plan-0",
            "carry_earlier": true,
            "inherit_from": ["plan-1", "plan-2"],
            "add_from": ["plan-3", "plan-4"],
        }),
    )]
    #[case::audit_words(
        &[
            "orgs",
            "audit",
            "acme",
            "--outcome",
            "refused,failed",
            "--actor",
            "user",
            "--actor",
            "host",
        ],
        json!({
            "org_id": "acme",
            "outcome": ["refused", "failed"],
            "actor": ["user", "host"],
        }),
    )]
    #[case::permissions(
        &["clients", "set-permissions", "ev1_a", "--permissions", "submit_results,perform_work"],
        json!({ "client_id": "ev1_a", "permissions": ["submit_results", "perform_work"] }),
    )]
    fn a_comma_list_repeats_or_splits_on_commas(
        #[case] argv: &[&str],
        #[case] expected: Value,
    ) -> anyhow::Result<()> {
        assert_eq!(Value::Object(parse(argv)?.args), expected);
        Ok(())
    }

    #[rstest]
    // `--base-plan-id` is required, so only `--add-from` can fail this.
    #[case::revise_takes_no_add_from(
        &["campaigns", "revise", "c", "--plan", "p", "--base-plan-id", "plan-0", "--add-from", "x"],
        clap::error::ErrorKind::UnknownArgument,
        "--add-from",
    )]
    #[case::each_audit_word_is_checked(
        &["audit", "list", "--outcome", "refused,denied"],
        clap::error::ErrorKind::InvalidValue,
        "denied",
    )]
    fn a_bad_list_flag_is_refused_by_clap(
        #[case] argv: &[&str],
        #[case] kind: clap::error::ErrorKind,
        #[case] named: &str,
    ) -> anyhow::Result<()> {
        let refused = parse(argv).err().context("took a bad list flag")?;
        let refused = refused
            .downcast_ref::<clap::Error>()
            .context("a clap error")?;
        assert_eq!(refused.kind(), kind);
        assert!(refused.to_string().contains(named), "{refused}");
        Ok(())
    }

    #[rstest]
    #[case::pairs(&["--labels", "study=lfm3", "--labels", "kind=a=b"], json!({ "study": "lfm3", "kind": "a=b" }))]
    #[case::json(&["--labels", r#"{"Team": "Edge ML"}"#], json!({ "Team": "Edge ML" }))]
    #[case::cleared(&["--labels", "{}"], json!({}))]
    #[case::empty_value(&["--labels", "kind="], json!({ "kind": "" }))]
    fn labels_parse_as_one_object(
        #[case] flags: &[&str],
        #[case] labels: Value,
    ) -> anyhow::Result<()> {
        let argv = ["campaigns", "rename", "campaign_1"]
            .into_iter()
            .chain(flags.iter().copied())
            .collect::<Vec<_>>();
        let command = parse(&argv)?;
        assert_eq!(
            Value::Object(command.args),
            json!({ "campaign_id": "campaign_1", "labels": labels })
        );
        Ok(())
    }

    #[rstest]
    #[case::no_pair(&["--labels", "study"], "takes KEY=VALUE or a JSON object")]
    #[case::repeated(&["--labels", "a=1", "--labels", "a=2"], "names `a` twice")]
    #[case::invalid_json(&["--labels", "{]"], "is not valid JSON")]
    fn bad_labels_are_refused(#[case] flags: &[&str], #[case] reason: &str) -> anyhow::Result<()> {
        let argv = ["campaigns", "rename", "campaign_1"]
            .into_iter()
            .chain(flags.iter().copied())
            .collect::<Vec<_>>();
        let refused = parse(&argv).err().context("took bad labels")?;
        let message = format!("{refused:#}");
        assert!(message.contains(reason), "{message}");
        Ok(())
    }

    #[test]
    fn an_account_is_typed_at_creation_and_by_its_own_verb() -> anyhow::Result<()> {
        let command = parse(&["users", "set-type", "user_1", "--account-type", "service"])?;
        assert_eq!(command.route.tool(), "users_set_type");
        assert_eq!(
            Value::Object(command.args),
            json!({ "user_id": "user_1", "account_type": "service" })
        );
        let command = parse(&[
            "users",
            "create",
            "--contact-email",
            "job@b.test",
            "--account-type",
            "service",
        ])?;
        assert_eq!(command.args.get("account_type"), Some(&json!("service")));
        Ok(())
    }

    #[rstest]
    #[case::set_type(&["users", "set-type", "user_1"], "--account-type")]
    #[case::set_status(&["users", "set-status", "user_1"], "--status")]
    fn an_account_change_requires_its_value(
        #[case] argv: &[&str],
        #[case] flag: &str,
    ) -> anyhow::Result<()> {
        let refused = parse(argv)
            .err()
            .with_context(|| format!("{argv:?} ran without {flag}"))?;
        let refused = refused
            .downcast_ref::<clap::Error>()
            .context("a clap error")?;
        assert_eq!(
            refused.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
        assert!(refused.to_string().contains(flag), "{refused}");
        Ok(())
    }

    #[test]
    fn a_verb_covered_elsewhere_is_not_offered_twice() {
        assert!(parse(&["users", "me"]).is_err());
    }

    /// A positional after a bare flag stays a positional.
    #[test]
    fn every_boolean_flag_reads_bare_as_true_and_takes_a_value() -> anyhow::Result<()> {
        let file = [
            "campaigns",
            "file",
            "--dry-run",
            "campaign_1",
            "--plan",
            "p.toml",
        ];
        let command = parse(&file)?;
        assert_eq!(
            Value::Object(command.args),
            json!({ "campaign_id": "campaign_1", "dry_run": true, "plan": "p.toml" })
        );
        let command = parse(&["vault", "download", "--overwrite", "vault_1", "--out", "a"])?;
        assert_eq!(
            Value::Object(command.args),
            json!({ "resource_id": "vault_1", "out": "a", "overwrite": true })
        );
        let command = parse(&[
            "campaigns",
            "file",
            "campaign_1",
            "--plan",
            "p",
            "--dry-run=false",
        ])?;
        assert_eq!(command.args.get("dry_run"), Some(&json!(false)));
        let command = parse(&["plans", "list", "--all=false"])?;
        assert_eq!(command.args.get("all"), Some(&json!(false)));
        Ok(())
    }

    #[test]
    fn fields_split_on_commas_and_repeat() -> anyhow::Result<()> {
        let command = parse(&[
            "orgs",
            "list",
            "--fields",
            "slug,org_id",
            "--fields",
            "status",
        ])?;
        assert_eq!(
            command.args.get("fields"),
            Some(&json!(["slug", "org_id", "status"]))
        );
        Ok(())
    }

    #[test]
    fn a_vault_file_takes_a_path_or_the_uploaded_object() -> anyhow::Result<()> {
        let command = parse(&[
            "vault",
            "upload",
            "--name",
            "n",
            "--type",
            "gguf_vision",
            "--model",
            r#"{"filename":"m.gguf","sha256":"ab"}"#,
            "--mmproj",
            "p.gguf",
        ])?;
        assert_eq!(
            command.args.get("model"),
            Some(&json!({ "filename": "m.gguf", "sha256": "ab" }))
        );
        assert_eq!(command.args.get("mmproj"), Some(&json!("p.gguf")));
        Ok(())
    }

    /// The invoker reads a definition, so it arrives as text, less any `@`.
    #[rstest]
    #[case::at_path("@b1.json", "b1.json")]
    #[case::path("b1.toml", "b1.toml")]
    #[case::json(r#"{"benchmark_type":"eval"}"#, r#"{"benchmark_type":"eval"}"#)]
    fn a_definition_reaches_the_invoker_as_text(
        #[case] given: &str,
        #[case] passed: &str,
    ) -> anyhow::Result<()> {
        let command = parse(&["orgs", "benchmark-put", "acme", "b1", "--definition", given])?;
        assert_eq!(command.args.get("definition"), Some(&json!(passed)));
        Ok(())
    }

    #[rstest]
    #[case::none_on_a_page(json!({ "unreadable": 0 }), None)]
    #[case::none_in_a_catalog(json!({ "unreadable_benchmark_ids": [] }), None)]
    #[case::counted(
        json!({ "unreadable": 2 }),
        Some("unreadable: 2 record(s) the server could not decode")
    )]
    #[case::named(
        json!({ "unreadable_benchmark_ids": ["b1", "b2"] }),
        Some("unreadable: b1,b2 (kept, but the server cannot read them)")
    )]
    fn a_listing_says_what_the_server_could_not_read(
        #[case] listing: Value,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let listing = listing.as_object().context("a listing is an object")?;
        assert_eq!(unreadable_line(listing).as_deref(), expected);
        Ok(())
    }

    /// Blocks are set off by blank lines; a list of one-line texts still joins.
    #[test]
    fn a_record_prints_each_multi_line_text_under_its_index() -> anyhow::Result<()> {
        let answer = json!({
            "base": { "plan_id": "plan_1" },
            "tags": ["a", "b"],
            "variants": ["[[variants]]\nmodel = \"a\"\n", "[[variants]]\nmodel = \"b\"\n"],
            "warnings": [],
        });
        let answer = answer.as_object().context("a record is an object")?;
        assert_eq!(
            record(answer),
            "base: plan_id=plan_1\n\
             tags: a,b\n\
             \n\
             # variants[0]\n\
             [[variants]]\n\
             model = \"a\"\n\
             \n\
             # variants[1]\n\
             [[variants]]\n\
             model = \"b\"\n\
             \n\
             warnings: (none)\n"
        );
        Ok(())
    }

    #[test]
    fn a_count_cannot_be_negative() {
        assert!(parse(&["clients", "list", "--limit", "-1"]).is_err());
    }

    /// `--adb-port` belongs to `run` and `kill`, the readiness flags to `run`.
    #[test]
    fn a_route_verb_takes_no_plan_runner_flag() {
        assert!(parse(&["campaigns", "list", "--adb-port", "5038"]).is_err());
        assert!(parse(&["campaigns", "list", "--readiness-skip-thermal"]).is_err());
    }

    #[test]
    fn a_table_cell_renders_nested_values_and_cuts_long_ones() {
        assert_eq!(render(&json!(["a", "b"])), "a,b");
        // Keys in sorted order, so the rendering is the same with or without
        // serde_json's `preserve_order`.
        assert_eq!(
            render(&json!({ "org_id": "org_1", "type": "org" })),
            "org_id=org_1,type=org"
        );
        let long = json!("f".repeat(64));
        let width = |column: &str, value: &Value| cell(column, value).chars().count();
        assert_eq!(width("public_key", &long), CELL_MAX_CHARS);
        assert_eq!(
            width("capabilities", &json!(vec!["flag"; 20])),
            CELL_MAX_CHARS
        );
        assert_eq!(width("client_id", &long), 64);
        assert_eq!(width("sha256", &long), 64);
    }

    /// An owner shows as its whole id, an object's later keys survive a long
    /// first one, and a value holding newlines stays on its row.
    #[test]
    fn a_table_cell_keeps_what_the_row_is_read_for() {
        let user = "user_01a0dc0d-e066-74d6-a724-5f0e9c3b2a18";
        assert_eq!(
            cell("owner", &json!({ "type": "user", "user_id": user })),
            user
        );
        let progress = json!({
            "computed_at": "2026-09-03T20:40:46.523914209Z",
            "counts": {
                "available": 0, "cancelled": 304, "failed": 0, "finished": 8, "running": 0,
                "total": 312,
            },
            "starved_groups": [],
        });
        assert_eq!(
            cell("progress_snapshot", &progress),
            "computed_at=2026-09-03T20:40:46.523914209Z,counts.available=0,\
             counts.cancelled=304,counts.failed=0,counts.finished=8,counts.running=0,\
             counts.total=312"
        );
        assert_eq!(
            cell(
                "client_details",
                &json!("{\n  \"platform\": \"android\"\n}")
            ),
            "{ \"platform\": \"android\" }"
        );
    }

    /// An object cell leaves out its ids and digests and puts `type` first; a
    /// wide one is cut at its budget.
    #[test]
    fn a_table_object_cell_stays_within_its_budget() {
        let file = |name: &str| {
            json!({
                "filename": name,
                "resource_id": "vault_01a0dc32-52fd-7523-a92f-b8bdda33788b",
                "sha256": "9c688eae".repeat(8),
            })
        };
        let artifact = json!({
            "mmproj": file("mmproj.gguf"),
            "model": file("model.gguf"),
            "type": "gguf_vision",
        });
        let dataset = json!({
            "adapter": "mmspec@1",
            "digest": "0fa0b6b6".repeat(8),
            "upstream": { "digest": "2bd3c7b4".repeat(8), "url": "github.com/x/MMSpec" },
        });
        assert_eq!(
            cell("artifact", &artifact),
            "type=gguf_vision,mmproj.filename=mmproj.gguf,model.filename=model.gguf"
        );
        assert_eq!(
            cell("dataset", &dataset),
            "adapter=mmspec@1,upstream.url=github.com/x/MMSpec"
        );
        let wide: Map<String, Value> = (0..8)
            .map(|n| (format!("count_{n}"), json!("x".repeat(40))))
            .collect();
        let text = cell("progress", &Value::Object(wide));
        assert_eq!(text.chars().count(), OBJECT_CELL_MAX_CHARS, "{text}");
        assert!(text.ends_with('…'), "{text}");
    }

    #[test]
    fn help_names_verbs_and_flags() -> anyhow::Result<()> {
        let route = |tool: &str| {
            ROUTES
                .iter()
                .find(|route| route.tool() == tool)
                .with_context(|| format!("no route {tool}"))
        };
        let upload = route_command(route("vault_upload")?);
        let about = upload.get_long_about().map(ToString::to_string);
        assert!(
            about.as_deref().is_some_and(
                |about| about.contains("`vault blob-put`") && about.contains("`--flavor`")
            ),
            "{about:?}"
        );
        let spellings = spellings(&route_command(route("vault_download")?));
        assert_eq!(
            respell(&spellings, "`resource_id`, `overwrite`, `users_me`"),
            "`RESOURCE_ID`, `--overwrite`, `auth me`"
        );
        Ok(())
    }

    /// `fields` narrows a read; a write has no `--fields` to fail after it ran.
    #[test]
    fn only_a_read_takes_fields() -> anyhow::Result<()> {
        let command = parse(&["orgs", "show", "acme", "--fields", "slug"])?;
        assert_eq!(command.args.get("fields"), Some(&json!(["slug"])));
        assert!(parse(&["orgs", "create", "--slug", "acme", "--fields", "slug"]).is_err());
        assert!(parse(&["orgs", "delete", "acme", "--fields", "org_id"]).is_err());
        Ok(())
    }

    /// On a terminal, a report rewrites the line of its file. On a pipe, each
    /// report is a separate line.
    #[rstest]
    #[case::terminal(
        Some((|| 200) as fn() -> usize),
        "importing org/m@1:a\n\
         \r  a: 5 of 10 bytes (50%)\x1b[K\
         \r  a: 10 of 10 bytes (100%)\x1b[K\n\
         importing org/v\n\
         \r  b: 1 of 4 bytes (25%)\x1b[K\n\
         importing org/w\n\
         \r  e: 3 of 4 bytes (75%)\x1b[K\n\
         \r  f: 0 of 0 bytes (100%)\x1b[K\n\
         \r  g: 1 of 2 bytes (50%)\x1b[K\n"
    )]
    #[case::pipe(
        None,
        "importing org/m@1:a\n  \
         a: 5 of 10 bytes (50%)\n  \
         a: 10 of 10 bytes (100%)\n\
         importing org/v\n  \
         b: 1 of 4 bytes (25%)\n\
         importing org/w\n  \
         e: 3 of 4 bytes (75%)\n  \
         f: 0 of 0 bytes (100%)\n  \
         g: 1 of 2 bytes (50%)\n"
    )]
    fn progress_lines_follow_the_output(
        #[case] columns: Option<fn() -> usize>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let mut out = Vec::new();
        let mut line = ProgressLine::new(&mut out, columns);
        [
            ("org/m@1:a", "a", 5, 10),
            ("org/m@1:a", "a", 10, 10),
            ("org/v", "b", 1, 4),
            ("org/w", "e", 3, 4),
            ("org/w", "f", 0, 0),
            ("org/w", "g", 1, 2),
        ]
        .into_iter()
        .for_each(|(hf_ref, file, bytes_done, bytes_total)| {
            line.report(&ImportProgress {
                hf_ref: hf_ref.into(),
                file: file.into(),
                bytes_done,
                bytes_total,
            })
        });
        line.finish();
        assert_eq!(String::from_utf8(out)?, expected);
        Ok(())
    }

    /// A progress line never passes the width it is given. A long file name
    /// keeps its end, which tells the shards of a model apart.
    #[rstest]
    #[case::fits("m.gguf", 80, "  m.gguf: 1 of 2 bytes (50%)")]
    #[case::long_name(
        "dir/model-00001-of-00004.safetensors",
        30,
        "  \u{2026}tensors: 1 of 2 bytes (50%)"
    )]
    #[case::wide_name("dir/模型权重文件.gguf", 30, "  \u{2026}件.gguf: 1 of 2 bytes (50%)")]
    #[case::narrow("m.gguf", 10, "  m.gguf: ")]
    fn a_progress_line_fits_the_terminal(
        #[case] file: &str,
        #[case] width: usize,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let line = fit(file, ": 1 of 2 bytes (50%)", width);
        assert_eq!(line, expected);
        assert!(line.width() <= width, "{line}");
        Ok(())
    }
}
