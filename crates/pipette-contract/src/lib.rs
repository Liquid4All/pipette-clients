//! Checks JSON fixtures against the serde types they encode: each example re-encodes
//! to itself and is named for the variants it holds, and every variant of every enum
//! an example holds has one. Variant lists come from serde, and enum names are found
//! by renaming each string and key. `fields` tries an example's fields one at a time,
//! and `ensure_current` keeps a file of what the crate derives from the examples, which a
//! client derives too. Names no wire type: `tests/contract.rs` runs it over this crate's
//! `fixtures` with `pipette-plan-types`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Serialize;
use serde::de::value::{MapDeserializer, StrDeserializer};
use serde::de::{self, DeserializeOwned};
use serde_json::Value;

mod fields;
#[cfg(test)]
mod tests;

pub use fields::{Bounds, Omitted, ReencodeFn, check_bounds, omittable};

const BLESS: &str = "CONTRACT_BLESS";

/// A variant name no enum on the wire has.
const UNKNOWN: &str = "\0";

/// The longest example file name. A Windows checkout fails on a path past 260
/// characters; the self-hosted runner's root and `crates/pipette-contract/fixtures/<section>/`
/// spend up to 115, and the rest leaves room for a longer section.
const MAX_FILE_NAME: usize = 120;

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Keeps the names serde lists for an unknown variant: every name the enum
/// accepts, aliases included.
#[derive(Debug)]
struct Accepted(Option<&'static [&'static str]>);

impl de::Error for Accepted {
    fn custom<T: fmt::Display>(_: T) -> Self {
        Self(None)
    }

    fn unknown_variant(_: &str, expected: &'static [&'static str]) -> Self {
        Self(Some(expected))
    }
}

impl fmt::Display for Accepted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::error::Error for Accepted {}

fn short_name<E>() -> &'static str {
    let path = std::any::type_name::<E>();
    path.rsplit("::").next().unwrap_or(path)
}

/// Every name the closed enum `E` accepts, when `tag` (or, if `None`, its bare
/// name) selects its variant.
fn accepted<E: DeserializeOwned>(tag: Option<&str>) -> anyhow::Result<&'static [&'static str]> {
    let probe = match tag {
        Some(tag) => E::deserialize(MapDeserializer::<_, Accepted>::new(
            [(tag, UNKNOWN)].into_iter(),
        )),
        None => E::deserialize(StrDeserializer::<Accepted>::new(UNKNOWN)),
    };
    let Err(Accepted(Some(names))) = probe else {
        anyhow::bail!(
            "{} is not a closed enum named by {tag:?}",
            short_name::<E>()
        );
    };
    Ok(names)
}

/// The crate's name for `value`'s variant, snake-cased from its derived `Debug`.
fn variant_name(value: &impl fmt::Debug) -> anyhow::Result<String> {
    let debug = format!("{value:?}");
    let camel = debug
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or_default();
    anyhow::ensure!(
        camel.starts_with(|c: char| c.is_ascii_uppercase()),
        "`{debug}` does not start with a variant name"
    );
    let mut snake = String::new();
    for c in camel.chars() {
        if c.is_ascii_uppercase() && !snake.is_empty() {
            snake.push('_');
        }
        snake.push(c.to_ascii_lowercase());
    }
    Ok(snake)
}

/// Every name `value` spells, a string or an object key, each with a copy of
/// `value` that has it renamed to `UNKNOWN`.
fn renamings(value: &Value) -> Vec<(String, Value)> {
    match value {
        Value::String(name) => vec![(name.clone(), UNKNOWN.into())],
        Value::Object(fields) => {
            let mut found = Vec::new();
            for (key, field) in fields {
                let mut renamed = fields.clone();
                renamed.remove(key);
                renamed.insert(UNKNOWN.to_owned(), field.clone());
                found.push((key.clone(), Value::Object(renamed)));
                for (name, field) in renamings(field) {
                    let mut renamed = fields.clone();
                    renamed.insert(key.clone(), field);
                    found.push((name, Value::Object(renamed)));
                }
            }
            found
        }
        Value::Array(items) => {
            let mut found = Vec::new();
            for (i, item) in items.iter().enumerate() {
                for (name, item) in renamings(item) {
                    let mut renamed = items.clone();
                    if let Some(slot) = renamed.get_mut(i) {
                        *slot = item;
                    }
                    found.push((name, Value::Array(renamed)));
                }
            }
            found
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Vec::new(),
    }
}

/// The names `authored` spells that `T` reads as a closed enum's variant: renamed,
/// each fails as an unknown variant. A validated string fails otherwise; an open set
/// (`LlamaCppFlavor`) decodes.
fn enum_names<T: DeserializeOwned>(authored: &Value) -> Vec<String> {
    let unknown_variant = format!("unknown variant `{UNKNOWN}`");
    renamings(authored)
        .into_iter()
        .filter_map(|(name, renamed)| {
            serde_json::from_value::<T>(renamed)
                .is_err_and(|err| err.to_string().starts_with(&unknown_variant))
                .then_some(name)
        })
        .collect()
}

/// `parts` as a file name spells them, joined by dots. A `/` cannot appear in a
/// file name, so it is written `_` (`vl-spec-decode-corpus/1` as `vl-spec-decode-corpus_1`).
fn file_name(parts: &[String]) -> String {
    parts.join(".").replace('/', "_")
}

/// Fails unless `file` is `<spelled>.json` or `<spelled>.<note>.json`, where
/// `spelled` is `parts` as `file_name` spells them.
fn ensure_named(file: &str, parts: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        file.len() <= MAX_FILE_NAME,
        "is {} characters, past the {MAX_FILE_NAME} a Windows checkout can hold: hold fewer variants",
        file.len()
    );
    let spelled = file_name(parts);
    let note = file
        .strip_suffix(".json")
        .and_then(|stem| stem.strip_prefix(spelled.as_str()));
    anyhow::ensure!(
        note.is_some_and(|note| note.is_empty() || (note.len() > 1 && note.starts_with('.'))),
        "holds {spelled}, so must be named {spelled}.json or {spelled}.<note>.json"
    );
    Ok(())
}

pub type NameFn<T> = fn(&T, &mut Name) -> anyhow::Result<()>;

pub fn reencode<T: Serialize + DeserializeOwned>(value: &Value) -> anyhow::Result<Value> {
    Ok(serde_json::to_value(serde_json::from_value::<T>(
        value.clone(),
    )?)?)
}

/// Every example on disk, by its `<section>/<file>` label.
pub fn examples() -> anyhow::Result<BTreeMap<String, PathBuf>> {
    let mut examples = BTreeMap::new();
    for dir in std::fs::read_dir(contract_dir())? {
        let dir = dir?.path();
        if dir.is_dir() {
            for example in std::fs::read_dir(dir)? {
                let path = example?.path();
                examples.insert(label(&path)?, path);
            }
        }
    }
    Ok(examples)
}

/// An example's `<section>/<file>` label, as `variants.json` and the derived files name it.
pub fn label(path: &Path) -> anyhow::Result<String> {
    let file = path.file_name().and_then(|name| name.to_str());
    let Some(file) = file else {
        anyhow::bail!("{} is no contract file", path.display());
    };
    Ok(format!("{}/{file}", section(path)?))
}

pub fn read_fixture<T: DeserializeOwned>(file: &str) -> anyhow::Result<T> {
    let path = contract_dir().join(file);
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    serde_json::from_str(&text).with_context(|| path.display().to_string())
}

/// Fails unless `fixtures/<file>` holds `current`; `CONTRACT_BLESS=1` rewrites it. Keys
/// are sorted, so the file reads the same whether or not `serde_json` keeps insertion order.
pub fn ensure_current(file: &str, current: &impl Serialize) -> anyhow::Result<()> {
    let path = contract_dir().join(file);
    let current = serde_json::to_string_pretty(&sorted(serde_json::to_value(current)?))? + "\n";
    if std::env::var_os(BLESS).is_some() {
        std::fs::write(&path, &current)?;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    anyhow::ensure!(
        committed == current,
        "{} is stale: rerun with {BLESS}=1 and commit it",
        path.display()
    );
    Ok(())
}

fn sorted(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let object: BTreeMap<_, _> = object.into_iter().collect();
            Value::Object(object.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        leaf @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => leaf,
    }
}

/// Checks one example with its section's name function.
pub type CheckFn = fn(&mut Contract, &Path) -> anyhow::Result<()>;

/// The section an example is in: the directory holding it.
pub fn section(path: &Path) -> anyhow::Result<&str> {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .with_context(|| format!("{} is in no section", path.display()))
}

/// `value`'s variant name: an enum of bare names encodes as one.
fn wire(value: &impl Serialize) -> anyhow::Result<String> {
    let encoded = serde_json::to_value(value)?;
    let name = encoded.as_str().context("encodes as no name")?;
    Ok(name.to_owned())
}

#[derive(Default)]
pub struct Contract {
    /// Enum, then variant, then the examples holding it: `variants.json`.
    held: BTreeMap<&'static str, BTreeMap<String, BTreeSet<String>>>,
    /// Enum, then each name it accepts that no example holds, with the parts it
    /// spells in a name. An alias is held by an example of the variant it names.
    unheld: BTreeMap<&'static str, BTreeMap<String, Vec<String>>>,
    /// Enum to the first example holding it and the name parts before it, which a
    /// missing variant's example can start from.
    first: BTreeMap<&'static str, (String, Vec<String>)>,
    /// Enum to its full type name, so two enums sharing a name cannot merge.
    types: BTreeMap<&'static str, &'static str>,
}

impl Contract {
    /// Every example on disk, each checked by `check`.
    pub fn read(check: CheckFn) -> anyhow::Result<Self> {
        let mut contract = Self::default();
        for path in examples()?.values() {
            check(&mut contract, path)?;
        }
        Ok(contract)
    }

    /// Fails with a line per variant no example holds, naming the file to add. The
    /// name stops at the missing variant; that file's own check names the rest.
    pub fn ensure_every_variant_held(&self) -> anyhow::Result<()> {
        let mut missing = String::new();
        for (name, unheld) in &self.unheld {
            for (variant, spelled) in unheld {
                missing += &format!("\n  {name}::{variant}");
                if let Some((example, before)) = self.first.get(name) {
                    let section = example.split('/').next().unwrap_or_default();
                    let parts = file_name(&[before.as_slice(), spelled].concat());
                    missing += &format!(": add {section}/{parts}.json");
                }
            }
        }
        anyhow::ensure!(
            missing.is_empty(),
            "{} has no example for{missing}",
            contract_dir().display()
        );
        Ok(())
    }

    /// Fails unless `variants.json` records what the examples hold;
    /// `CONTRACT_BLESS=1` rewrites it.
    pub fn ensure_variants_json_current(&self) -> anyhow::Result<()> {
        ensure_current("variants.json", &self.held)
    }

    /// Checks the example at `path`, `<section>/<name>.json`, where `name` spells
    /// the variants it holds.
    pub fn file<T: Serialize + DeserializeOwned>(
        &mut self,
        path: &Path,
        name: NameFn<T>,
    ) -> anyhow::Result<()> {
        let label = label(path)?;
        let file = label.rsplit('/').next().unwrap_or_default();
        self.named_example(&label, file, path, name)
            .with_context(|| label.clone())
    }

    fn named_example<T: Serialize + DeserializeOwned>(
        &mut self,
        label: &str,
        file: &str,
        path: &Path,
        name: NameFn<T>,
    ) -> anyhow::Result<()> {
        let authored: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let parts = self.example(label, &authored, name)?;
        ensure_named(file, &parts)
    }

    /// Checks that `authored` re-encodes to itself and that `name` spells exactly the
    /// enum names it reads, recording the variants it holds. Returns the name parts.
    fn example<T: Serialize + DeserializeOwned>(
        &mut self,
        label: &str,
        authored: &Value,
        name: NameFn<T>,
    ) -> anyhow::Result<Vec<String>> {
        let value: T = serde_json::from_value(authored.clone())?;
        // `Model` drops unknown keys, so this is what catches a mistyped one.
        let encoded = serde_json::to_value(&value)?;
        anyhow::ensure!(
            &encoded == authored,
            "does not re-encode to itself: {encoded}"
        );
        let mut spelling = Name {
            contract: self,
            example: label,
            parts: Vec::new(),
        };
        name(&value, &mut spelling)?;
        let parts = spelling.parts;
        let mut enums = enum_names::<T>(authored);
        let mut spelled = parts.clone();
        enums.sort();
        spelled.sort();
        anyhow::ensure!(
            enums == spelled,
            "reads {enums:?} as enum variants, but its name spells {spelled:?}"
        );
        Ok(parts)
    }
}

/// Spells an example's name from the variants it holds, recording each.
pub struct Name<'a> {
    contract: &'a mut Contract,
    example: &'a str,
    parts: Vec<String>,
}

/// One field of the key that picks a `keyed` enum's variant.
pub struct Key {
    field: &'static str,
    name: String,
    accepted: &'static [&'static str],
}

impl Key {
    /// `value` must encode as a bare name.
    pub fn new<E: Serialize + DeserializeOwned>(
        field: &'static str,
        value: &E,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            field,
            name: wire(value)?,
            accepted: accepted::<E>(None)?,
        })
    }
}

impl Name<'_> {
    /// The variant `value` holds, `E` being a closed enum that encodes as a bare
    /// name (`VllmFlavor`) or as an object tagged by one of its fields (`Model`).
    pub fn holds<E>(&mut self, value: &E) -> anyhow::Result<()>
    where
        E: Serialize + DeserializeOwned + PartialEq,
    {
        let encoded = serde_json::to_value(value)?;
        // The tag is the field that alone fails as an unknown variant.
        let tag = encoded
            .as_object()
            .and_then(|fields| fields.keys().find(|key| accepted::<E>(Some(key)).is_ok()))
            .map(String::as_str);
        let accepted = accepted::<E>(tag)?;
        let name = self.register::<E>(|| {
            Ok(accepted
                .iter()
                .map(|&n| (n.to_owned(), vec![n.to_owned()]))
                .collect())
        })?;
        // `value` with its variant renamed to `variant`.
        let renamed = |variant: &str| -> Value {
            match (tag, encoded.clone()) {
                (Some(tag), Value::Object(mut fields)) => {
                    fields.insert(tag.to_owned(), variant.into());
                    Value::Object(fields)
                }
                _ => variant.into(),
            }
        };
        let variant = match tag {
            Some(tag) => encoded.get(tag),
            None => Some(&encoded),
        }
        .and_then(Value::as_str)
        .with_context(|| format!("{name} does not encode its variant's name"))?;
        // An alias decodes to the same value.
        self.hold(name, variant, vec![variant.to_owned()], |n| {
            serde_json::from_value::<E>(renamed(n)).ok().as_ref() == Some(value)
        });
        Ok(())
    }

    /// A `try_from` enum picked by `keys` together, as `ModelFlags` is: its variants are the
    /// key combinations that decode, and an example's name spells its keys.
    pub fn keyed<E>(&mut self, value: &E, keys: &[Key]) -> anyhow::Result<()>
    where
        E: DeserializeOwned + fmt::Debug + strum::EnumCount,
    {
        let name = self.register::<E>(|| {
            let combinations = keys.iter().fold(
                vec![(serde_json::Map::new(), Vec::<String>::new())],
                |combinations, key| {
                    combinations
                        .iter()
                        .flat_map(|(fields, spelled)| {
                            key.accepted.iter().map(move |&n| {
                                let mut fields = fields.clone();
                                fields.insert(key.field.to_owned(), n.into());
                                let mut spelled = spelled.clone();
                                spelled.push(n.to_owned());
                                (fields, spelled)
                            })
                        })
                        .collect()
                },
            );
            let variants = combinations
                .into_iter()
                .filter_map(|(fields, spelled)| {
                    let variant = serde_json::from_value::<E>(fields.into()).ok()?;
                    Some(variant_name(&variant).map(|name| (name, spelled)))
                })
                .collect::<anyhow::Result<BTreeMap<_, _>>>()?;
            anyhow::ensure!(
                variants.len() == E::COUNT,
                "{} of {}'s {} variants decode from a combination of its keys",
                variants.len(),
                short_name::<E>(),
                E::COUNT
            );
            Ok(variants)
        })?;
        let variant = variant_name(value)?;
        let spelled = keys.iter().map(|key| key.name.clone()).collect();
        self.hold(name, &variant, spelled, |n| n == variant);
        Ok(())
    }

    /// `E`'s short name, registering `variants` the first time, and this example
    /// as where it is held if it comes first; fails when another enum took the name.
    fn register<E>(
        &mut self,
        variants: impl FnOnce() -> anyhow::Result<BTreeMap<String, Vec<String>>>,
    ) -> anyhow::Result<&'static str> {
        let name = short_name::<E>();
        let full = std::any::type_name::<E>();
        let known = *self.contract.types.entry(name).or_insert(full);
        anyhow::ensure!(
            known == full,
            "two enums are named {name}: {known} and {full}"
        );
        if !self.contract.unheld.contains_key(name) {
            self.contract.unheld.insert(name, variants()?);
        }
        let here = (self.example.to_owned(), self.parts.clone());
        let first = self
            .contract
            .first
            .entry(name)
            .or_insert_with(|| here.clone());
        if here.0 < first.0 {
            *first = here;
        }
        Ok(name)
    }

    /// Records that the example holds `name`'s `variant`, with every name `same`
    /// picks, and appends `spelled` to its name.
    fn hold(
        &mut self,
        name: &'static str,
        variant: &str,
        spelled: Vec<String>,
        same: impl Fn(&str) -> bool,
    ) {
        self.contract
            .held
            .entry(name)
            .or_default()
            .entry(variant.to_owned())
            .or_default()
            .insert(self.example.to_owned());
        if let Some(unheld) = self.contract.unheld.get_mut(name) {
            unheld.retain(|n, _| !same(n));
        }
        self.parts.extend(spelled);
    }
}
