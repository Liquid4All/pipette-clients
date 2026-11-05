//! An example's fields one at a time: which a payload may leave out, and which values
//! each may carry. A field is named by its JSON pointer, reached through objects only;
//! a field inside an array (an eval's samples) is not tried.

use std::collections::BTreeMap;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Decodes a value as its section's type and encodes it back.
pub type ReencodeFn = fn(&Value) -> anyhow::Result<Value>;

/// What a field an example holds reads as once a payload leaves it out.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Omitted {
    Absent,
    Default(Value),
}

/// Values tried in place of one field: each `accepted` one reads back as itself, and
/// each `refused` one fails to decode.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    #[serde(default)]
    pub accepted: Vec<Value>,
    #[serde(default)]
    pub refused: Vec<Value>,
}

/// Each field of `authored` a payload may leave out, with what it then reads as; any
/// other is required. Fails when an omission changes more than its own field.
pub fn omittable(
    authored: &Value,
    reencode: ReencodeFn,
) -> anyhow::Result<BTreeMap<String, Omitted>> {
    let mut pointers = Vec::new();
    fields(authored, "", &mut pointers);
    let mut omittable = BTreeMap::new();
    for pointer in pointers {
        let without = without(authored, &pointer)?;
        let Ok(read) = reencode(&without) else {
            continue;
        };
        let omitted = if read == without {
            Omitted::Absent
        } else {
            let default = read.pointer(&pointer).cloned();
            match default {
                Some(default) if with(&without, &pointer, default.clone())? == read => {
                    Omitted::Default(default)
                }
                _ => anyhow::bail!("without {pointer}, re-encodes as {read}"),
            }
        };
        omittable.insert(pointer, omitted);
    }
    Ok(omittable)
}

/// Checks each value `bounds` tries at `pointer`, which must name a field of `authored`.
pub fn check_bounds(
    authored: &Value,
    pointer: &str,
    bounds: &Bounds,
    reencode: ReencodeFn,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        authored.pointer(pointer).is_some(),
        "{pointer} names no field of the example"
    );
    for value in &bounds.accepted {
        let tried = with(authored, pointer, value.clone())?;
        let read = reencode(&tried).with_context(|| format!("{pointer} = {value} must decode"))?;
        anyhow::ensure!(read == tried, "{pointer} = {value} re-encodes as {read}");
    }
    for value in &bounds.refused {
        let tried = with(authored, pointer, value.clone())?;
        anyhow::ensure!(
            reencode(&tried).is_err(),
            "{pointer} = {value} must be refused, but decodes"
        );
    }
    Ok(())
}

fn fields(value: &Value, at: &str, found: &mut Vec<String>) {
    if let Value::Object(object) = value {
        for (key, field) in object {
            let pointer = format!("{at}/{}", key.replace('~', "~0").replace('/', "~1"));
            found.push(pointer.clone());
            fields(field, &pointer, found);
        }
    }
}

/// The object holding `pointer`'s field in `value`, and the field's key.
fn parent<'v>(
    value: &'v mut Value,
    pointer: &str,
) -> anyhow::Result<(&'v mut serde_json::Map<String, Value>, String)> {
    let (at, key) = pointer
        .rsplit_once('/')
        .with_context(|| format!("{pointer} is no pointer"))?;
    let object = value
        .pointer_mut(at)
        .and_then(Value::as_object_mut)
        .with_context(|| format!("{pointer} is in no object"))?;
    Ok((object, key.replace("~1", "/").replace("~0", "~")))
}

fn without(value: &Value, pointer: &str) -> anyhow::Result<Value> {
    let mut value = value.clone();
    let (object, key) = parent(&mut value, pointer)?;
    object.remove(&key);
    Ok(value)
}

fn with(value: &Value, pointer: &str, field: Value) -> anyhow::Result<Value> {
    let mut value = value.clone();
    let (object, key) = parent(&mut value, pointer)?;
    object.insert(key, field);
    Ok(value)
}
