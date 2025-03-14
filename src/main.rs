mod config;
mod image_processing;
mod transformations;
mod types;

use clap::{Arg, Command};
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("HashTest Generator")
        .version("1.0")
        .author("Your Name")
        .about("Generates image variants for testing perceptual hashing algorithms")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("INPUT_DIR")
                .help("Directory containing original images (defaults to current directory)")
                .required(false),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Directory for output image variants (defaults to 'variants' subfolder in current directory)")
                .required(false),
        )
        .arg(
            Arg::new("recursive")
                .short('r')
                .long("recursive")
                .help("Search for images recursively in input directory"),
        )
        .arg(
            Arg::new("force")
                .short('f')
                .long("force")
                .help("Force overwrite of existing output files"),
        )
        .arg(
            Arg::new("stirmark")
                .long("stirmark")
                .value_name("STIRMARK_PATH")
                .help("Path to StirMark utility executable"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("CONFIG_FILE")
                .help("Configuration file for transformations (defaults to config.json in current directory)")
                .required(false),
        )
        .get_matches();

    // Get current directory
    let current_dir = std::env::current_dir().expect("Failed to get current directory");

    // Extract command-line arguments
    let input_dir = match matches.get_one::<String>("input") {
        Some(input) => PathBuf::from(input),
        None => current_dir.clone(),
    };

    let output_dir = match matches.get_one::<String>("output") {
        Some(output) => PathBuf::from(output),
        None => current_dir.join("variants"), // Create a subfolder called 'variants'
    };

    let recursive = matches.contains_id("recursive");
    let force_overwrite = matches.contains_id("force");
    let stirmark_path = matches
        .get_one::<String>("stirmark")
        .map(|s| PathBuf::from(s));

    // Get config file path - using explicit String instead of &str to avoid borrowing issues
    let config_path = if let Some(config_file) = matches.get_one::<String>("config") {
        // User specified a config file
        Some(config_file.as_str())
    } else {
        // Check for default config in current directory
        let default_config = current_dir.join("config.json");
        if default_config.exists() {
            // Just pass None and let the config module handle the default
            None
        } else {
            None
        }
    };

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Create configuration
    let config = config::create_config(
        input_dir,
        output_dir,
        recursive,
        stirmark_path,
        config_path,
        force_overwrite,
    )?;

    // Process the images
    image_processing::process_images(config)?;

    Ok(())
}
