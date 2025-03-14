use crate::transformations::default_transformations;
use crate::types::{Config, Transformation};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn parse_config_file(path: &str) -> Result<Vec<Transformation>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut transformations = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() != 2 {
            continue;
        }

        let transform_type = parts[0].trim();
        let params = parts[1].trim();

        match transform_type {
            "resize" => {
                let values: Vec<&str> = params.split(',').collect();
                if values.len() == 2 {
                    if let (Ok(width), Ok(height)) =
                        (values[0].parse::<u32>(), values[1].parse::<u32>())
                    {
                        transformations.push(Transformation::Resize(width, height));
                    }
                }
            }
            "rotate" => {
                if let Ok(angle) = params.parse::<f32>() {
                    transformations.push(Transformation::Rotate(angle));
                }
            }
            "flip_horizontal" => {
                if params == "true" {
                    transformations.push(Transformation::Flip(true));
                }
            }
            "flip_vertical" => {
                if params == "true" {
                    transformations.push(Transformation::Flip(false));
                }
            }
            "crop" => {
                let values: Vec<&str> = params.split(',').collect();
                if values.len() == 4 {
                    if let (Ok(x), Ok(y), Ok(width), Ok(height)) = (
                        values[0].parse::<u32>(),
                        values[1].parse::<u32>(),
                        values[2].parse::<u32>(),
                        values[3].parse::<u32>(),
                    ) {
                        transformations.push(Transformation::Crop(x, y, width, height));
                    }
                }
            }
            "blur" => {
                if let Ok(sigma) = params.parse::<f32>() {
                    transformations.push(Transformation::Blur(sigma));
                }
            }
            "noise" => {
                if let Ok(amount) = params.parse::<f32>() {
                    transformations.push(Transformation::Noise(amount));
                }
            }
            "compression" => {
                if let Ok(quality) = params.parse::<u8>() {
                    transformations.push(Transformation::Compression(quality));
                }
            }
            "brightness" => {
                if let Ok(factor) = params.parse::<f32>() {
                    transformations.push(Transformation::Brightness(factor));
                }
            }
            "contrast" => {
                if let Ok(factor) = params.parse::<f32>() {
                    transformations.push(Transformation::Contrast(factor));
                }
            }
            _ => {
                println!("Unknown transformation: {}", transform_type);
            }
        }
    }

    Ok(transformations)
}

pub fn create_config(
    input_dir: PathBuf,
    output_dir: PathBuf,
    recursive: bool,
    stirmark_path: Option<PathBuf>,
    config_path: Option<&str>,
    force_overwrite: bool,
) -> Result<Config, Box<dyn Error>> {
    // Set up default transformations or read from config file
    let transformations = if let Some(config_path) = config_path {
        parse_config_file(config_path)?
    } else {
        default_transformations()
    };

    Ok(Config {
        input_dir,
        output_dir,
        transformations,
        recursive,
        stirmark_path,
        force_overwrite,
    })
}
