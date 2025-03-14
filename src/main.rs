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
                .help("Directory containing original images")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Directory for output image variants")
                .required(true),
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
                .help("Configuration file for transformations"),
        )
        .get_matches();

    // Extract command-line arguments
    let input_dir = PathBuf::from(matches.get_one::<String>("input").unwrap());
    let output_dir = PathBuf::from(matches.get_one::<String>("output").unwrap());
    let recursive = matches.contains_id("recursive");
    let force_overwrite = matches.contains_id("force");
    let stirmark_path = matches
        .get_one::<String>("stirmark")
        .map(|s| PathBuf::from(s));

    // Testing paths - Comment out in production
    let input_dir = PathBuf::from("/Users/richardlyon/Desktop/test-images/original_images");
    let output_dir = PathBuf::from("/Users/richardlyon/Desktop/test-images/variant_images");

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Create configuration
    let config = config::create_config(
        input_dir,
        output_dir,
        recursive,
        stirmark_path,
        matches.get_one::<String>("config").map(|s| s.as_str()),
        force_overwrite,
    )?;

    // Process the images
    image_processing::process_images(config)?;

    Ok(())
}
