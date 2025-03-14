pub mod config;
pub mod image_processing;
pub mod transformations;
pub mod types;

// Re-export commonly used items
pub use config::create_config;
pub use image_processing::process_images;
pub use types::{Config, SupportedFormat, Transformation};
