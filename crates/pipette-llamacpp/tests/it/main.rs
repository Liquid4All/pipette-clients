//! One test binary for all of this crate's integration tests: Windows links each binary slowly.

mod common;

mod asr_audio_tokens;
mod asr_eval_corpus;
mod asr_max_memory_smoke;
mod asr_throughput_chat_endpoint;
mod end_to_end_latency_cells;
mod eval_apply_template;
mod eval_connection_reset_recovery;
mod eval_crash_recovery;
mod eval_parallel;
mod eval_parallel_crash;
mod eval_parallel_transport_failure;
mod eval_sampling;
mod eval_stop_reason;
mod llama_bench_cells;
mod vl_chat_error_body;
mod vl_max_memory_smoke;
mod vl_spec_decode_cells;
mod vl_spec_decode_corpus;
mod vl_throughput_chat_endpoint;
mod vl_throughput_helper_placement;
mod vl_throughput_image_tokens;
mod vl_throughput_text_prompt;
mod vlm_eval_broken_stream;
mod vlm_eval_corpus;
mod vlm_eval_error_body;
mod vlm_eval_gate_then_resume;
mod vlm_eval_parallel;
mod vlm_eval_parallel_crash;
mod vlm_eval_parallel_transport_failure;
mod vlm_eval_reasoning;
mod vlm_eval_reasoning_restart;
mod vlm_eval_sampling;
mod vlm_eval_serial_is_unchanged;
