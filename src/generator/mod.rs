//! Generator configuration types for LLM interactions

mod info;
mod params;

pub use info::GeneratorInfo;
pub use params::{CompletionParameters, NodeCompletionParameters, ProviderSettings};
