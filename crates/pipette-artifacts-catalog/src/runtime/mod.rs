//! The runtime catalogs. All ship bundled except [`llamacpp`], which lists
//! upstream releases from GitHub, and [`trt_edgellm`], which has no catalog,
//! only its unpacked layout.

pub mod llamacpp;
pub mod mlx;
pub mod onnx_genai;
pub mod openvino;
pub mod ovms;
pub mod torch_oai;
pub mod trt_edgellm;
