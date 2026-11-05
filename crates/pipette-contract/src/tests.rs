//! The engine's checks on made-up types, so a refactor cannot quietly weaken them.

use serde::Deserialize;
use serde_json::json;

use super::*;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Shape {
    Circle {
        label: String,
        color: Color,
    },
    #[serde(alias = "box")]
    Square {
        side: u32,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Color {
    Red,
    Blue,
}

/// A map keyed by an enum, which no name function spells.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Palette {
    counts: BTreeMap<Color, u32>,
}

mod other {
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    pub(super) enum Color {
        Green,
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Pair {
    first: Color,
    second: other::Color,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Finish {
    Matte,
    Gloss,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Base {
    Water,
    Oil,
}

/// A `try_from` enum as `RuntimeFlags` is: `color`, `finish` and `base` select the variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, strum::EnumCount)]
#[serde(try_from = "PaintRef", into = "PaintRef")]
enum Paint {
    RedMatte,
    BlueGloss,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PaintRef {
    color: Color,
    finish: Finish,
    base: Base,
}

impl TryFrom<PaintRef> for Paint {
    type Error = String;

    fn try_from(paint: PaintRef) -> Result<Self, Self::Error> {
        match (paint.color, paint.finish, paint.base) {
            (Color::Red, Finish::Matte, Base::Water) => Ok(Self::RedMatte),
            (Color::Blue, Finish::Gloss, Base::Oil) => Ok(Self::BlueGloss),
            _ => Err("no such paint".to_owned()),
        }
    }
}

impl From<Paint> for PaintRef {
    fn from(paint: Paint) -> Self {
        match paint {
            Paint::RedMatte => Self {
                color: Color::Red,
                finish: Finish::Matte,
                base: Base::Water,
            },
            Paint::BlueGloss => Self {
                color: Color::Blue,
                finish: Finish::Gloss,
                base: Base::Oil,
            },
        }
    }
}

fn paint_name(paint: &Paint, name: &mut Name) -> anyhow::Result<()> {
    let flat = PaintRef::from(paint.clone());
    name.keyed(
        paint,
        &[
            Key::new("color", &flat.color)?,
            Key::new("finish", &flat.finish)?,
            Key::new("base", &flat.base)?,
        ],
    )
}

fn shape_name(shape: &Shape, name: &mut Name) -> anyhow::Result<()> {
    name.holds(shape)?;
    match shape {
        Shape::Circle { color, .. } => name.holds(color),
        Shape::Square { .. } => Ok(()),
    }
}

fn circle(color: &str) -> Value {
    json!({"type": "circle", "label": "c", "color": color})
}

fn square() -> Value {
    json!({"type": "square", "side": 4})
}

fn strings(names: &[&str]) -> Vec<String> {
    names.iter().map(ToString::to_string).collect()
}

fn contract<T: Serialize + DeserializeOwned>(
    examples: &[Value],
    name: NameFn<T>,
) -> anyhow::Result<Contract> {
    let mut contract = Contract::default();
    for (i, authored) in examples.iter().enumerate() {
        contract.example(&format!("shape/{i}.json"), authored, name)?;
    }
    Ok(contract)
}

fn failure<T: Serialize + DeserializeOwned>(
    examples: &[Value],
    name: NameFn<T>,
) -> anyhow::Result<String> {
    let err = contract(examples, name).err().context("should fail")?;
    Ok(format!("{err:#}"))
}

#[test]
fn a_name_spells_the_variants_an_example_holds() -> anyhow::Result<()> {
    let mut contract = Contract::default();
    let parts = contract.example("shape/circle.red.json", &circle("red"), shape_name)?;
    assert_eq!(parts, strings(&["circle", "red"]));
    assert!(
        contract
            .held
            .get("Color")
            .is_some_and(|held| held.contains_key("red"))
    );

    ensure_named("circle.red.json", &parts)?;
    ensure_named("circle.red.thin.json", &parts)?;
    for wrong in [
        "circle.blue.json",
        "circle.json",
        "circle.redder.json",
        "circle.red..json",
    ] {
        assert!(ensure_named(wrong, &parts).is_err(), "{wrong}");
    }
    Ok(())
}

#[test]
fn a_slash_in_a_variant_is_spelled_as_an_underscore() -> anyhow::Result<()> {
    let parts = strings(&["vl-spec-decode-corpus/1", "mmspec@1"]);
    ensure_named("vl-spec-decode-corpus_1.mmspec@1.json", &parts)?;
    assert!(ensure_named("vl-spec-decode-corpus/1.mmspec@1.json", &parts).is_err());
    Ok(())
}

#[rstest::rstest]
#[case::at_the_cap(MAX_FILE_NAME, None)]
#[case::one_past(MAX_FILE_NAME + 1, Some("Windows checkout"))]
fn a_file_name_a_windows_checkout_cannot_hold_fails(
    #[case] len: usize,
    #[case] refusal: Option<&str>,
) -> anyhow::Result<()> {
    let parts = vec!["x".repeat(len - ".json".len())];
    let named = ensure_named(&format!("{}.json", parts[0]), &parts);
    match refusal {
        None => named?,
        Some(words) => anyhow::ensure!(
            named.is_err_and(|err| err.to_string().contains(words)),
            "{len} characters must be refused"
        ),
    }
    Ok(())
}

#[test]
fn a_variant_is_missing_until_an_example_holds_it() -> anyhow::Result<()> {
    let one = contract(&[circle("red")], shape_name)?;
    let unheld = |name: &str| -> Vec<String> {
        one.unheld
            .get(name)
            .map(|names| names.keys().cloned().collect())
            .unwrap_or_default()
    };
    assert_eq!(unheld("Shape"), strings(&["box", "square"]));
    assert_eq!(unheld("Color"), strings(&["blue"]));

    let err = one
        .ensure_every_variant_held()
        .err()
        .context("should fail")?
        .to_string();
    assert!(
        err.contains("Color::blue: add shape/circle.blue.json"),
        "{err}"
    );
    assert!(
        err.contains("Shape::square: add shape/square.json"),
        "{err}"
    );

    let all = contract(&[circle("red"), circle("blue"), square()], shape_name)?;
    all.ensure_every_variant_held()
}

#[test]
fn a_keyed_variant_is_each_combination_that_decodes() -> anyhow::Result<()> {
    let mut contract = Contract::default();
    let red_matte = json!({"color": "red", "finish": "matte", "base": "water"});
    let parts = contract.example("paint/red.matte.water.json", &red_matte, paint_name)?;
    assert_eq!(parts, strings(&["red", "matte", "water"]));
    assert!(
        contract
            .held
            .get("Paint")
            .is_some_and(|held| held.contains_key("red_matte"))
    );
    assert!(
        !contract.held.contains_key("Color") && !contract.unheld.contains_key("Color"),
        "a key is no enum of its own"
    );

    let err = contract
        .ensure_every_variant_held()
        .err()
        .context("should fail")?
        .to_string();
    assert!(
        err.contains("Paint::blue_gloss: add paint/blue.gloss.oil.json"),
        "{err}"
    );
    Ok(())
}

#[test]
fn an_enum_the_name_leaves_out_is_found_by_renaming() -> anyhow::Result<()> {
    let skips = |shape: &Shape, name: &mut Name| name.holds(shape);
    let err = failure(&[circle("red")], skips)?;
    assert!(
        err.contains(r#"reads ["circle", "red"] as enum variants"#),
        "{err}"
    );

    let spells_nothing = |_: &Palette, _: &mut Name| Ok(());
    let err = failure(&[json!({"counts": {"blue": 1}})], spells_nothing)?;
    assert!(err.contains(r#"reads ["blue"] as enum variants"#), "{err}");
    Ok(())
}

#[test]
fn a_plain_string_is_not_an_enum() -> anyhow::Result<()> {
    let spells_label = |shape: &Shape, name: &mut Name| {
        shape_name(shape, name)?;
        name.holds(&Color::Red)
    };
    let err = failure(
        &[json!({"type": "circle", "label": "red", "color": "red"})],
        spells_label,
    )?;
    assert!(
        err.contains(r#"reads ["circle", "red"] as enum variants, but its name spells ["circle", "red", "red"]"#),
        "{err}"
    );
    Ok(())
}

#[test]
fn two_enums_sharing_a_name_fail() -> anyhow::Result<()> {
    let both = |pair: &Pair, name: &mut Name| {
        name.holds(&pair.first)?;
        name.holds(&pair.second)
    };
    let err = failure(&[json!({"first": "red", "second": "Green"})], both)?;
    assert!(err.contains("two enums are named Color"), "{err}");
    Ok(())
}

/// A field of each kind `omittable` tells apart, and a map whose key needs escaping.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Knobs {
    name: String,
    #[serde(default = "three")]
    level: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    #[serde(default)]
    inner: Inner,
    counts: BTreeMap<String, u8>,
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
struct Inner {
    #[serde(default)]
    on: bool,
}

fn three() -> u8 {
    3
}

/// Omitting `a` resets `b` too.
#[derive(Debug, Serialize, Deserialize)]
#[serde(from = "RawLinked")]
struct Linked {
    a: u8,
    b: u8,
}

#[derive(Deserialize)]
struct RawLinked {
    a: Option<u8>,
    b: u8,
}

impl From<RawLinked> for Linked {
    fn from(raw: RawLinked) -> Self {
        match raw.a {
            Some(a) => Self { a, b: raw.b },
            None => Self { a: 0, b: 0 },
        }
    }
}

#[test]
fn a_field_is_required_absent_or_defaulted() -> anyhow::Result<()> {
    let authored = json!({
        "name": "a", "level": 5, "note": "n", "inner": {"on": true}, "counts": {"x/y": 1},
    });
    let omittable = omittable(&authored, reencode::<Knobs>)?;
    let expected = BTreeMap::from([
        ("/counts/x~1y".to_owned(), Omitted::Absent),
        ("/inner".to_owned(), Omitted::Default(json!({"on": false}))),
        ("/inner/on".to_owned(), Omitted::Default(json!(false))),
        ("/level".to_owned(), Omitted::Default(json!(3))),
        ("/note".to_owned(), Omitted::Absent),
    ]);
    assert_eq!(omittable, expected);
    Ok(())
}

#[test]
fn an_omission_that_changes_another_field_fails() {
    let err = omittable(&json!({"a": 1, "b": 7}), reencode::<Linked>);
    assert!(
        err.is_err_and(|err| err.to_string().starts_with("without /a, re-encodes as")),
        "an omission that resets /b must fail"
    );
}

#[rstest::rstest]
#[case::within(&[json!(0), json!(255)], &[json!(-1), json!(256)], None)]
#[case::accepted_is_refused(&[json!(256)], &[], Some("/level = 256 must decode"))]
#[case::refused_is_accepted(&[], &[json!(255)], Some("/level = 255 must be refused, but decodes"))]
fn a_bound_is_checked_both_ways(
    #[case] accepted: &[Value],
    #[case] refused: &[Value],
    #[case] failure: Option<&str>,
) -> anyhow::Result<()> {
    let authored = json!({"name": "a", "level": 5, "inner": {"on": true}, "counts": {}});
    let bounds = Bounds {
        accepted: accepted.to_vec(),
        refused: refused.to_vec(),
    };
    let checked = check_bounds(&authored, "/level", &bounds, reencode::<Knobs>);
    match failure {
        None => checked?,
        Some(words) => anyhow::ensure!(
            checked.is_err_and(|err| err.to_string() == words),
            "must fail with {words}"
        ),
    }
    Ok(())
}

#[test]
fn a_bound_on_no_field_fails() {
    let authored = json!({"name": "a", "counts": {}});
    let bounds = Bounds::default();
    let checked = check_bounds(&authored, "/level", &bounds, reencode::<Knobs>);
    assert!(checked.is_err_and(|err| err.to_string() == "/level names no field of the example"));
}
