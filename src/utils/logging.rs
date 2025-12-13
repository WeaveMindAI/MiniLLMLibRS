//! Logging configuration

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Log level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Configure logging for the library
///
/// This sets up tracing with a subscriber that outputs to stdout.
/// The log level can be overridden by the RUST_LOG environment variable.
///
/// # Example
/// ```no_run
/// use minillmlib::utils::configure_logging;
/// use minillmlib::utils::LogLevel;
///
/// configure_logging(LogLevel::Debug);
/// ```
pub fn configure_logging(level: LogLevel) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("minillmlib={}", level.as_str())));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_level(true))
        .with(filter)
        .try_init()
        .ok(); // Ignore error if already initialized
}

/// Configure logging with a custom filter string
///
/// # Example
/// ```no_run
/// use minillmlib::utils::logging::configure_logging_with_filter;
///
/// configure_logging_with_filter("minillmlib=debug,reqwest=warn");
/// ```
pub fn configure_logging_with_filter(filter: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_level(true))
        .with(filter)
        .try_init()
        .ok();
}
