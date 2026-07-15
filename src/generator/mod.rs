//! Generator configuration types for LLM interactions

mod info;
mod params;
pub(crate) mod pricing;

pub use info::GeneratorInfo;
pub use params::{
    CompletionParameters, NodeCompletionParameters, ProviderSettings, ReasoningConfig,
};
pub use pricing::ModelRates;
