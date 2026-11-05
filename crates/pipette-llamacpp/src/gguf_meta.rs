//! A projector's GGUF metadata, read before a server starts: whether it holds
//! an audio encoder, which one, and whether a vision encoder sits beside it,
//! which tells two models sharing one audio encoder apart.
//!
//! Only the metadata block, and for a file that could be OpenBMB's the tensor
//! names after it, is read, and every value the query does not need is skipped
//! unread, so a projector whose metadata carries a vocabulary costs a few seek
//! calls rather than its size.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

use anyhow::Context;

/// Metadata keys a real file holds are a few hundred; a count past this is a
/// corrupt or hostile header, not a model.
const MAX_KV_COUNT: u64 = 1 << 20;
/// No key or projector name is longer, and a string past it is read as data to skip.
const MAX_KEPT_STRING: u64 = 1 << 16;

const HAS_AUDIO_ENCODER: &str = "clip.has_audio_encoder";
const HAS_VISION_ENCODER: &str = "clip.has_vision_encoder";
const AUDIO_PROJECTOR_TYPE: &str = "clip.audio.projector_type";
const PROJECTOR_TYPE: &str = "clip.projector_type";
const ARCHITECTURE: &str = "general.architecture";

/// The audio encoder OpenBMB's own MiniCPM-o 4.5 audio GGUF is
/// (`audio/MiniCPM-o-4_5-audio-F16.gguf` in `openbmb/MiniCPM-o-4_5-gguf`):
/// written for OpenBMB's runtime as `general.architecture = whisper`, with no
/// `clip.*` keys. Pipette's ASR build of llama.cpp recognizes that file by its
/// architecture, its `audio_pool_step` key and its
/// `audio_projector.linear2.weight` tensor, and loads it as this encoder.
pub const OPENBMB_MINICPMO_AUDIO_ENCODER: &str = "minicpmoa";
const WHISPER_ARCHITECTURE: &str = "whisper";
const AUDIO_POOL_STEP: &str = "audio_pool_step";
const OPENBMB_PROJECTOR_TENSOR: &str = "audio_projector.linear2.weight";
/// A projector holds a few thousand tensors at most; a count past this is a
/// corrupt header, and its names are not read.
const MAX_TENSOR_COUNT: u64 = 1 << 16;

/// What a projector says about its audio encoder.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectorMetadata {
    /// `clip.has_audio_encoder`; absent reads as false.
    pub has_audio_encoder: bool,
    /// `clip.has_vision_encoder`; absent reads as false. Qwen3-Omni's projector
    /// holds its vision tower beside the `qwen3a` audio encoder, Qwen3-ASR's
    /// only the audio encoder (llama.cpp's `conversion/qwen3vl.py`).
    pub has_vision_encoder: bool,
    /// `clip.audio.projector_type`: the audio encoder of a file that keeps one
    /// key per modality (Qwen-Omni, Gemma).
    pub audio_projector_type: Option<String>,
    /// `clip.projector_type`: the encoder of a legacy single-encoder file.
    pub projector_type: Option<String>,
    /// The file is OpenBMB's whisper-architecture MiniCPM-o audio GGUF, which
    /// carries no `clip.*` key ([`OPENBMB_MINICPMO_AUDIO_ENCODER`]).
    pub openbmb_minicpmo_audio: bool,
}

impl ProjectorMetadata {
    pub fn holds_audio_encoder(&self) -> bool {
        self.has_audio_encoder || self.openbmb_minicpmo_audio
    }

    pub fn audio_encoder_name(&self) -> Option<&str> {
        self.audio_projector_type
            .as_deref()
            .or(self.projector_type.as_deref())
            .or_else(|| {
                self.openbmb_minicpmo_audio
                    .then_some(OPENBMB_MINICPMO_AUDIO_ENCODER)
            })
    }
}

pub fn projector_metadata(path: &Path) -> anyhow::Result<ProjectorMetadata> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_projector_metadata(BufReader::new(file))
        .with_context(|| format!("reading the metadata of {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl TryFrom<u32> for ValueType {
    type Error = anyhow::Error;

    fn try_from(tag: u32) -> anyhow::Result<Self> {
        Ok(match tag {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => anyhow::bail!("a metadata value of unknown type {other}"),
        })
    }
}

impl ValueType {
    /// The bytes one value takes, for the types whose size is fixed.
    fn width(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// The tensor and metadata entry counts, after the magic and version.
fn read_counts<R: Read>(reader: &mut R) -> anyhow::Result<(u64, u64)> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"GGUF", "not a GGUF file");
    let version = read_u32(reader)?;
    anyhow::ensure!(
        (2..=3).contains(&version),
        "GGUF version {version} is not one this reader knows (2 and 3)"
    );
    let tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;
    anyhow::ensure!(
        kv_count <= MAX_KV_COUNT,
        "{kv_count} metadata entries: not a model's header"
    );
    Ok((tensor_count, kv_count))
}

/// The metadata entry count, after the magic, version and tensor count.
fn read_header<R: Read>(reader: &mut R) -> anyhow::Result<u64> {
    Ok(read_counts(reader)?.1)
}

fn read_projector_metadata<R: Read + Seek>(mut reader: R) -> anyhow::Result<ProjectorMetadata> {
    let (tensor_count, kv_count) = read_counts(&mut reader)?;

    let mut found = ProjectorMetadata::default();
    let mut architecture = None;
    let mut audio_pool_step = false;
    for _ in 0..kv_count {
        let key = read_string(&mut reader, MAX_KEPT_STRING)?
            .context("a metadata key longer than any key is")?;
        let value_type = ValueType::try_from(read_u32(&mut reader)?)?;
        match (key.as_str(), value_type) {
            (HAS_AUDIO_ENCODER, ValueType::Bool) => {
                let mut byte = [0_u8; 1];
                reader.read_exact(&mut byte)?;
                found.has_audio_encoder = byte[0] != 0;
            }
            (HAS_VISION_ENCODER, ValueType::Bool) => {
                let mut byte = [0_u8; 1];
                reader.read_exact(&mut byte)?;
                found.has_vision_encoder = byte[0] != 0;
            }
            (AUDIO_PROJECTOR_TYPE, ValueType::String) => {
                found.audio_projector_type = read_string(&mut reader, MAX_KEPT_STRING)?;
            }
            (PROJECTOR_TYPE, ValueType::String) => {
                found.projector_type = read_string(&mut reader, MAX_KEPT_STRING)?;
            }
            (ARCHITECTURE, ValueType::String) => {
                architecture = read_string(&mut reader, MAX_KEPT_STRING)?;
            }
            (AUDIO_POOL_STEP, _) => {
                audio_pool_step = true;
                skip_value(&mut reader, value_type)?;
            }
            _ => skip_value(&mut reader, value_type)?,
        }
    }
    // Recognized as the ASR build recognizes it: the tensor names follow the
    // metadata, and are read only for a file that could be OpenBMB's.
    if architecture.as_deref() == Some(WHISPER_ARCHITECTURE)
        && audio_pool_step
        && tensor_count <= MAX_TENSOR_COUNT
    {
        found.openbmb_minicpmo_audio =
            has_tensor(&mut reader, tensor_count, OPENBMB_PROJECTOR_TENSOR)?;
    }
    Ok(found)
}

/// Whether the tensor infos that follow the metadata name `wanted`: each is
/// a name, a dimension count, the dimensions, a type and an offset.
fn has_tensor<R: Read + Seek>(reader: &mut R, count: u64, wanted: &str) -> anyhow::Result<bool> {
    for _ in 0..count {
        let name = read_string(reader, MAX_KEPT_STRING)?;
        if name.as_deref() == Some(wanted) {
            return Ok(true);
        }
        let dims = read_u32(reader)?;
        skip(
            reader,
            u64::from(dims).saturating_mul(8).saturating_add(4 + 8),
        )?;
    }
    Ok(false)
}

const TOKENS: &str = "tokenizer.ggml.tokens";

/// The pieces the vocabulary of the GGUF model at `path` gives `ids`, in
/// order, each `None` past the vocabulary's end. `None` when the file has no
/// vocabulary.
pub fn token_pieces(path: &Path, ids: &[u32]) -> anyhow::Result<Option<Vec<Option<String>>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_token_pieces(BufReader::new(file), ids)
        .with_context(|| format!("reading the vocabulary of {}", path.display()))
}

/// [`token_pieces`] over a reader. The vocabulary is read only as far as the
/// largest id asked for, every other piece skipped; nothing after it is read.
fn read_token_pieces<R: Read + Seek>(
    mut reader: R,
    ids: &[u32],
) -> anyhow::Result<Option<Vec<Option<String>>>> {
    let kv_count = read_header(&mut reader)?;
    for _ in 0..kv_count {
        let key = read_string(&mut reader, MAX_KEPT_STRING)?
            .context("a metadata key longer than any key is")?;
        let value_type = ValueType::try_from(read_u32(&mut reader)?)?;
        if key != TOKENS || value_type != ValueType::Array {
            skip_value(&mut reader, value_type)?;
            continue;
        }
        let element_type = ValueType::try_from(read_u32(&mut reader)?)?;
        anyhow::ensure!(
            element_type == ValueType::String,
            "{TOKENS} holds {element_type:?}, not strings"
        );
        let len = read_u64(&mut reader)?;
        let mut pieces = vec![None; ids.len()];
        let last = ids.iter().copied().max().map_or(0, u64::from);
        for index in 0..len.min(last.saturating_add(1)) {
            if ids.iter().any(|&id| u64::from(id) == index) {
                let piece = read_string(&mut reader, MAX_KEPT_STRING)?;
                for (slot, _) in pieces
                    .iter_mut()
                    .zip(ids)
                    .filter(|(_, id)| u64::from(**id) == index)
                {
                    slot.clone_from(&piece);
                }
            } else {
                skip_value(&mut reader, ValueType::String)?;
            }
        }
        return Ok(Some(pieces));
    }
    Ok(None)
}

fn skip_value<R: Read + Seek>(reader: &mut R, value_type: ValueType) -> anyhow::Result<()> {
    match value_type.width() {
        Some(width) => skip(reader, width),
        None if value_type == ValueType::String => {
            let len = read_u64(reader)?;
            skip(reader, len)
        }
        None => {
            let element_type = ValueType::try_from(read_u32(reader)?)?;
            let len = read_u64(reader)?;
            // llama.cpp refuses nested arrays too; skipping them would recurse once
            // per level, so a crafted file could overflow the stack.
            anyhow::ensure!(element_type != ValueType::Array, "a nested metadata array");
            match element_type.width() {
                Some(width) => skip(reader, len.saturating_mul(width)),
                // Strings are as long as they say.
                None => (0..len).try_for_each(|_| skip_value(reader, element_type)),
            }
        }
    }
}

/// Moves `bytes` forward. A seek past the end is legal for a file, so a header
/// that lies about a length is caught by the next read, not here.
fn skip<R: Seek>(reader: &mut R, bytes: u64) -> anyhow::Result<()> {
    let bytes = i64::try_from(bytes).context("a metadata value longer than any file")?;
    reader.seek_relative(bytes)?;
    Ok(())
}

/// A length-prefixed string, or `None` (and the string skipped) if it is past `max`.
fn read_string<R: Read + Seek>(reader: &mut R, max: u64) -> anyhow::Result<Option<String>> {
    let len = read_u64(reader)?;
    if len > max {
        skip(reader, len)?;
        return Ok(None);
    }
    let mut bytes = vec![0_u8; usize::try_from(len)?];
    reader.read_exact(&mut bytes)?;
    Ok(Some(
        String::from_utf8(bytes).context("a metadata string that is not UTF-8")?,
    ))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// GGUF headers for tests: OpenBMB's audio GGUF, a projector naming its audio
/// encoder, a model with a vocabulary, or any typed entries.
#[cfg(test)]
pub(crate) mod fixture {
    pub(crate) const TYPE_UINT32: u32 = 4;
    pub(crate) const TYPE_BOOL: u32 = 7;
    pub(crate) const TYPE_STRING: u32 = 8;
    pub(crate) const TYPE_ARRAY: u32 = 9;

    /// A GGUF v3 header with the given entries: `(key, type, encoded value)`.
    pub(crate) fn gguf(entries: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3_u32.to_le_bytes());
        out.extend(0_u64.to_le_bytes());
        out.extend((entries.len() as u64).to_le_bytes());
        for (key, value_type, value) in entries {
            out.extend((key.len() as u64).to_le_bytes());
            out.extend(key.as_bytes());
            out.extend(value_type.to_le_bytes());
            out.extend(value);
        }
        out
    }

    pub(crate) fn string(value: &str) -> Vec<u8> {
        let mut out = (value.len() as u64).to_le_bytes().to_vec();
        out.extend(value.as_bytes());
        out
    }

    /// An array of strings, as `tokenizer.ggml.tokens` stores its pieces.
    pub(crate) fn strings<S: AsRef<str>>(pieces: &[S]) -> Vec<u8> {
        let mut out = TYPE_STRING.to_le_bytes().to_vec();
        out.extend((pieces.len() as u64).to_le_bytes());
        for piece in pieces {
            out.extend(string(piece.as_ref()));
        }
        out
    }

    /// A projector whose audio encoder is `encoder`.
    pub(crate) fn projector(encoder: &str) -> Vec<u8> {
        gguf(&[
            ("general.architecture", TYPE_STRING, string("clip")),
            (super::HAS_AUDIO_ENCODER, TYPE_BOOL, vec![1]),
            (super::AUDIO_PROJECTOR_TYPE, TYPE_STRING, string(encoder)),
        ])
    }

    /// OpenBMB's MiniCPM-o audio GGUF, its metadata and tensor names only: each
    /// of `tensors` is one-dimensional.
    pub(crate) fn openbmb_audio(pool_step: bool, tensors: &[&str]) -> Vec<u8> {
        let mut entries = vec![
            ("general.architecture", TYPE_STRING, string("whisper")),
            (
                "general.description",
                TYPE_STRING,
                string("audio encoder for MiniCPM-omni"),
            ),
            ("d_model", TYPE_UINT32, 1024_u32.to_le_bytes().to_vec()),
        ];
        if pool_step {
            entries.push(("audio_pool_step", TYPE_UINT32, 5_u32.to_le_bytes().to_vec()));
        }
        let mut out = gguf(&entries);
        // The tensor count sits after the magic and the version.
        if let Some(count) = out.get_mut(8..16) {
            count.copy_from_slice(&(tensors.len() as u64).to_le_bytes());
        }
        for name in tensors {
            out.extend(string(name));
            out.extend(1_u32.to_le_bytes());
            out.extend(1024_u64.to_le_bytes());
            out.extend(1_u32.to_le_bytes());
            out.extend(0_u64.to_le_bytes());
        }
        out
    }

    /// A model whose vocabulary is `len` placeholder pieces, `pieces` at
    /// their ids.
    pub(crate) fn model(len: u32, pieces: &[(u32, &str)]) -> Vec<u8> {
        let vocabulary: Vec<String> = (0..len)
            .map(|id| {
                pieces
                    .iter()
                    .find(|(at, _)| *at == id)
                    .map_or_else(|| format!("t{id}"), |(_, piece)| (*piece).to_owned())
            })
            .collect();
        gguf(&[
            ("general.architecture", TYPE_STRING, string("qwen3")),
            (super::TOKENS, TYPE_ARRAY, strings(&vocabulary)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::fixture::{TYPE_ARRAY, TYPE_BOOL, TYPE_STRING, TYPE_UINT32, gguf, string, strings};
    use super::*;

    const TYPE_UINT16: u32 = 2;
    const TYPE_FLOAT32: u32 = 6;
    const TYPE_FLOAT64: u32 = 12;

    fn read(bytes: Vec<u8>) -> anyhow::Result<ProjectorMetadata> {
        read_projector_metadata(Cursor::new(bytes))
    }

    #[test]
    fn the_projector_keys_are_found_among_values_of_every_type() -> anyhow::Result<()> {
        let mut vocabulary = TYPE_STRING.to_le_bytes().to_vec();
        vocabulary.extend(3_u64.to_le_bytes());
        for token in ["<s>", "hello", "world"] {
            vocabulary.extend(string(token));
        }
        let mut floats = TYPE_FLOAT32.to_le_bytes().to_vec();
        floats.extend(2_u64.to_le_bytes());
        floats.extend([0_u8; 8]);
        let bytes = gguf(&[
            ("general.architecture", TYPE_STRING, string("clip")),
            (HAS_AUDIO_ENCODER, TYPE_BOOL, vec![1]),
            (HAS_VISION_ENCODER, TYPE_BOOL, vec![1]),
            (
                "clip.audio.block_count",
                TYPE_UINT32,
                32_u32.to_le_bytes().to_vec(),
            ),
            ("tokenizer.ggml.tokens", TYPE_ARRAY, vocabulary),
            ("clip.audio.mean", TYPE_ARRAY, floats),
            ("clip.audio.epsilon", TYPE_FLOAT64, vec![0; 8]),
            (AUDIO_PROJECTOR_TYPE, TYPE_STRING, string("qwen3a")),
            (PROJECTOR_TYPE, TYPE_STRING, string("qwen2.5o")),
        ]);
        let got = read(bytes)?;
        assert_eq!(
            got,
            ProjectorMetadata {
                has_audio_encoder: true,
                has_vision_encoder: true,
                audio_projector_type: Some("qwen3a".to_owned()),
                projector_type: Some("qwen2.5o".to_owned()),
                openbmb_minicpmo_audio: false,
            }
        );
        assert_eq!(got.audio_encoder_name(), Some("qwen3a"));
        Ok(())
    }

    #[test]
    fn a_missing_or_mistyped_key_is_absent() -> anyhow::Result<()> {
        let bytes = gguf(&[
            (PROJECTOR_TYPE, TYPE_UINT32, 7_u32.to_le_bytes().to_vec()),
            (HAS_AUDIO_ENCODER, TYPE_UINT32, 1_u32.to_le_bytes().to_vec()),
            (
                HAS_VISION_ENCODER,
                TYPE_UINT32,
                1_u32.to_le_bytes().to_vec(),
            ),
        ]);
        assert_eq!(read(bytes)?, ProjectorMetadata::default());
        Ok(())
    }

    #[test]
    fn a_nested_array_is_refused() {
        let mut nested = TYPE_ARRAY.to_le_bytes().to_vec();
        nested.extend(1_u64.to_le_bytes());
        nested.extend(TYPE_UINT16.to_le_bytes());
        nested.extend(2_u64.to_le_bytes());
        nested.extend([0_u8; 4]);
        let bytes = gguf(&[("odd.nesting", TYPE_ARRAY, nested)]);
        let err = read(bytes).err().map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("a nested metadata array"));
    }

    #[test]
    fn an_unknown_value_type_is_refused() {
        let bytes = gguf(&[("k", 13, vec![0; 8])]);
        let err = read(bytes).err().map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("a metadata value of unknown type 13"));
    }

    #[test]
    fn a_file_that_is_not_gguf_is_refused() {
        let err = read(b"RIFF....".to_vec()).err().map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("not a GGUF file"));
    }

    #[test]
    fn a_header_claiming_absurd_metadata_is_refused() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(u64::MAX.to_le_bytes());
        let err = read(bytes).err().map(|e| e.to_string());
        assert!(err.is_some_and(|e| e.contains("not a model's header")));
    }

    /// A file cut short is an error, not a partial answer read as complete.
    #[test]
    fn a_truncated_header_is_an_error() {
        let mut bytes = gguf(&[(AUDIO_PROJECTOR_TYPE, TYPE_STRING, string("qwen3a"))]);
        bytes.truncate(bytes.len() - 3);
        assert!(read(bytes).is_err());
    }

    /// A whisper file without the pool step or the projector tensor, as another
    /// whisper encoder would be, is not OpenBMB's.
    #[rstest::rstest]
    #[case::openbmb(true, &["encoder.conv1.weight", "audio_projector.linear2.weight"], true)]
    #[case::no_pool_step(false, &["audio_projector.linear2.weight"], false)]
    #[case::no_projector(true, &["encoder.conv1.weight"], false)]
    fn openbmbs_audio_gguf_is_the_minicpmoa_encoder(
        #[case] pool_step: bool,
        #[case] tensors: &[&str],
        #[case] openbmb: bool,
    ) -> anyhow::Result<()> {
        let got = read(super::fixture::openbmb_audio(pool_step, tensors))?;
        assert_eq!(got.openbmb_minicpmo_audio, openbmb, "{got:?}");
        assert_eq!(got.holds_audio_encoder(), openbmb, "{got:?}");
        assert_eq!(
            got.audio_encoder_name(),
            openbmb.then_some(OPENBMB_MINICPMO_AUDIO_ENCODER)
        );
        Ok(())
    }

    /// The published file itself, where a copy is at hand.
    #[test]
    fn the_published_minicpmo_audio_gguf_is_recognized() -> anyhow::Result<()> {
        let Some(path) = std::env::var_os("PIPETTE_TEST_MINICPMO_AUDIO_GGUF") else {
            return Ok(());
        };
        let got = projector_metadata(Path::new(&path))?;
        assert!(got.openbmb_minicpmo_audio, "{path:?}: {got:?}");
        assert_eq!(got.audio_encoder_name(), Some("minicpmoa"));
        Ok(())
    }

    /// A real projector, where one is at hand.
    #[test]
    fn a_real_projector_names_its_audio_encoder() -> anyhow::Result<()> {
        let Some(path) = std::env::var_os("PIPETTE_TEST_AUDIO_MMPROJ") else {
            return Ok(());
        };
        let got = projector_metadata(Path::new(&path))?;
        assert!(got.has_audio_encoder, "{path:?}: {got:?}");
        assert!(
            got.audio_encoder_name().is_some_and(|n| !n.is_empty()),
            "no projector type in {path:?}: {got:?}"
        );
        Ok(())
    }

    /// The pieces asked for come back by id, in the order asked, after other
    /// entries; an id past the vocabulary is `None`.
    #[test]
    fn token_pieces_are_read_by_id() -> anyhow::Result<()> {
        let bytes = gguf(&[
            ("general.architecture", TYPE_STRING, string("qwen3")),
            (
                "tokenizer.ggml.merges",
                TYPE_ARRAY,
                strings(&["a b", "c d"]),
            ),
            (TOKENS, TYPE_ARRAY, strings(&["<s>", "hello", "<|im_end|>"])),
        ]);
        let got = read_token_pieces(Cursor::new(bytes), &[2, 0, 7])?;
        assert_eq!(
            got,
            Some(vec![
                Some("<|im_end|>".to_owned()),
                Some("<s>".to_owned()),
                None
            ])
        );
        Ok(())
    }

    /// A file with no vocabulary says so, and a vocabulary of numbers is
    /// refused.
    #[test]
    fn a_missing_or_mistyped_vocabulary_is_reported() -> anyhow::Result<()> {
        let none = gguf(&[("general.architecture", TYPE_STRING, string("clip"))]);
        assert_eq!(read_token_pieces(Cursor::new(none), &[0])?, None);
        let mut numbers = TYPE_UINT32.to_le_bytes().to_vec();
        numbers.extend(1_u64.to_le_bytes());
        numbers.extend(7_u32.to_le_bytes());
        let mistyped = gguf(&[(TOKENS, TYPE_ARRAY, numbers)]);
        assert!(read_token_pieces(Cursor::new(mistyped), &[0]).is_err());
        Ok(())
    }
}
