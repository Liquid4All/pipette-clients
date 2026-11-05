//! The pure half of `build.rs`'s vendored llama.cpp tag derivation.
//!
//! `src/lib.rs` also includes it under `cfg(test)`: a build script is not a test
//! target, and a mistake here only shows as an APK advertising the wrong tag.

/// What `vendor/llama.cpp.tag` records, judged against the pin.
#[derive(Debug, PartialEq, Eq)]
pub enum Recorded<'a> {
    /// A release tag, recorded against the commit the submodule is on.
    Tag(&'a str),
    /// Fewer than the two whitespace-separated fields the format requires.
    Malformed,
    /// A first field that is not a release tag.
    NotATag(&'a str),
    /// A well-formed entry that has fallen behind the pin.
    Stale { tag: &'a str, sha: &'a str },
}

/// The tag `contents` records, checked against `head_sha`.
///
/// Only the first line is read, matching `ios/build-llama.sh`'s `read -r`: both
/// derivations must agree on which tag a pin carries.
pub fn recorded_tag<'a>(contents: &'a str, head_sha: &str) -> Recorded<'a> {
    let mut fields = contents
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace();
    let (Some(tag), Some(sha)) = (fields.next(), fields.next()) else {
        return Recorded::Malformed;
    };
    if !is_llamacpp_tag(tag) {
        return Recorded::NotATag(tag);
    }
    // A wrong tag would advertise a build this APK does not contain, so a
    // recording that has fallen behind the pin is dropped rather than trusted.
    if sha != head_sha {
        return Recorded::Stale { tag, sha };
    }
    Recorded::Tag(tag)
}

/// `bNNNN`, llama.cpp's release tag shape. Checked rather than assumed: a branch
/// name pointing at the same commit is not a build id.
pub fn is_llamacpp_tag(value: &str) -> bool {
    matches!(value.strip_prefix('b'), Some(digits)
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const HEAD: &str = "876a4321163249c43ca4e986818fab5ab081f282";

    #[rstest]
    #[case::on_the_pin(
        "b10216 876a4321163249c43ca4e986818fab5ab081f282\n",
        Recorded::Tag("b10216")
    )]
    #[case::no_trailing_newline(
        "b10216 876a4321163249c43ca4e986818fab5ab081f282",
        Recorded::Tag("b10216")
    )]
    #[case::extra_fields_ignored(
        "b10216 876a4321163249c43ca4e986818fab5ab081f282 b10217\n",
        Recorded::Tag("b10216")
    )]
    // Otherwise a bumped pin could be tagged from a leftover line.
    #[case::later_lines_ignored(
        "b10216 876a4321163249c43ca4e986818fab5ab081f282\nb9999 0f0f0f0f\n",
        Recorded::Tag("b10216")
    )]
    #[case::empty("", Recorded::Malformed)]
    #[case::tag_without_sha("b10216\n", Recorded::Malformed)]
    #[case::leading_blank_line(
        "\nb10216 876a4321163249c43ca4e986818fab5ab081f282\n",
        Recorded::Malformed
    )]
    #[case::branch_name(
        "master 876a4321163249c43ca4e986818fab5ab081f282",
        Recorded::NotATag("master")
    )]
    // Bumped submodule, forgotten recording.
    #[case::behind_the_pin("b10216 0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f", Recorded::Stale { tag: "b10216", sha: "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f" })]
    fn reads_the_recording(#[case] contents: &str, #[case] expected: Recorded<'_>) {
        assert_eq!(recorded_tag(contents, HEAD), expected);
    }

    #[rstest]
    #[case("b10216", true)]
    #[case("b0", true)]
    #[case("master", false)]
    #[case("876a43211", false)]
    #[case("b", false)]
    #[case("b10216-dirty", false)]
    // Lowercasing is the Kotlin side's job (`Capabilities`), not this check's.
    #[case("B10216", false)]
    #[case("", false)]
    fn recognizes_the_tag_shape(#[case] value: &str, #[case] expected: bool) {
        assert_eq!(is_llamacpp_tag(value), expected);
    }
}
