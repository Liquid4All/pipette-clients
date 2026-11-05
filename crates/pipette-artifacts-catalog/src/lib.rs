//! What pipette can install: the runtime catalogs the CLI lists and resolves
//! `--runtime` URIs through, and the OVMS pins and unpacked layout that
//! `pipette-artifacts` installs against and `pipette-ovms` launches from, and
//! the TensorRT Edge-LLM layout `pipette-trt-edgellm` launches from.

mod version_catalog;

pub mod runtime;
