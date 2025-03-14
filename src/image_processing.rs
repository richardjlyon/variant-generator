use crate::transformations::{apply_transformation, transformation_name};
use crate::types::{Config, SupportedFormat};
use image::DynamicImage;
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

pub fn process_images(config: Config) -> Result<(), Box<dyn Error>> {
    let entries = if config.recursive {
        find_images_recursive(&config.input_dir)?
    } else {
        find_images(&config.input_dir)?
    };

    if entries.is_empty() {
        return Ok(());
    }

    // Use Arc to share the Config across threads
    let config = Arc::new(config);

    // Collect any errors that occur during parallel processing
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));

    // Process images in parallel
    entries.par_iter().for_each(|entry| {
        let config = Arc::clone(&config);
        let errors = Arc::clone(&errors);

        if let Err(e) = process_image(entry, &config) {
            let mut errors = errors.lock().unwrap();
            errors.push(format!("Error processing {}: {}", entry.display(), e));
        }
    });

    // Check if any errors occurred during parallel processing
    let locked_errors = errors.lock().unwrap();
    if !locked_errors.is_empty() {
        for error in locked_errors.iter() {
            eprintln!("{}", error);
        }
        return Err(format!("Failed to process {} images", locked_errors.len()).into());
    }

    // If StirMark is available, use it for additional transformations
    if let Some(stirmark_path) = &config.stirmark_path {
        use_stirmark(&config.input_dir, &config.output_dir, stirmark_path)?;
    }

    Ok(())
}

pub fn process_image(img_path: &Path, config: &Arc<Config>) -> Result<(), Box<dyn Error>> {
    let filename = img_path.file_name().unwrap().to_str().unwrap();
    let stem = Path::new(filename).file_stem().unwrap().to_str().unwrap();

    let ext = img_path.extension().unwrap().to_str().unwrap();
    let format = SupportedFormat::from_extension(ext).unwrap();

    // Create a subfolder for this image's variants
    let image_subfolder = config.output_dir.join(stem);
    fs::create_dir_all(&image_subfolder)?;

    // Load the image
    let img = match format {
        SupportedFormat::HEIC => {
            // For HEIC, we need to convert it first using libheif-rs
            #[cfg(feature = "heic")]
            {
                convert_heic_to_dynamic_image(img_path)?
            }
            #[cfg(not(feature = "heic"))]
            {
                return Ok(());
            }
        }
        _ => image::open(img_path)?,
    };

    // Apply each transformation and save the result
    for (i, transform) in config.transformations.iter().enumerate() {
        let transform_name = transformation_name(transform);

        // Generate output filename
        let output_filename = format!("{}_{}_{}.{}", stem, transform_name, i, ext);
        let output_path = image_subfolder.join(&output_filename);

        // Skip if the file already exists (useful for resuming interrupted operations)
        if output_path.exists() && !config.force_overwrite {
            continue;
        }

        // Apply the transformation
        let transformed = apply_transformation(&img, transform, &format)?;

        // Save the transformed image
        save_image(&transformed, &output_path, format)?;
    }

    Ok(())
}

pub fn find_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut results = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if let Some(_) = SupportedFormat::from_extension(ext.to_str().unwrap_or("")) {
                    results.push(path);
                }
            }
        }
    }
    Ok(results)
}

pub fn find_images_recursive(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut results = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            results.extend(find_images_recursive(&path)?);
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                if let Some(_) = SupportedFormat::from_extension(ext.to_str().unwrap_or("")) {
                    results.push(path);
                }
            }
        }
    }

    Ok(results)
}

pub fn save_image(
    img: &DynamicImage,
    path: &Path,
    format: SupportedFormat,
) -> Result<(), Box<dyn Error>> {
    match format {
        SupportedFormat::HEIC => {
            #[cfg(feature = "heic")]
            {
                save_as_heic(img, path)?;
            }
            #[cfg(not(feature = "heic"))]
            {
                println!(
                    "HEIC support not compiled in. Saving as PNG instead: {}",
                    path.display()
                );
                img.save(path.with_extension("png"))?;
            }
        }
        _ => {
            if let Some(image_format) = format.to_image_format() {
                // Convert to RGB8 for JPEG format
                if format == SupportedFormat::JPEG {
                    let rgb8 = img.to_rgb8();
                    DynamicImage::ImageRgb8(rgb8).save_with_format(path, image_format)?;
                } else {
                    img.save_with_format(path, image_format)?;
                }
            } else {
                img.save(path)?;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "heic")]
fn convert_heic_to_dynamic_image(path: &Path) -> Result<DynamicImage, Box<dyn Error>> {
    use libheif_rs::{ColorSpace, HeifContext};

    let context = HeifContext::read_from_file(path)?;
    let handle = context.primary_image_handle()?;
    let image = handle.decode(ColorSpace::Rgb, None)?;

    let width = image.width();
    let height = image.height();
    let stride = image.stride(0)?;
    let data = image.planes().interleaved;

    // Convert to RGB8 format that image crate can understand
    let img = DynamicImage::ImageRgb8(
        image::ImageBuffer::from_raw(width as u32, height as u32, data.to_vec()).unwrap(),
    );

    Ok(img)
}

#[cfg(feature = "heic")]
fn save_as_heic(img: &DynamicImage, path: &Path) -> Result<(), Box<dyn Error>> {
    use libheif_rs::{HeifChroma, HeifColorspace, HeifContext, HeifEncoderQuality};

    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let mut context = HeifContext::new()?;
    let mut encoder = context.encoder_for_format(libheif_rs::HeifCompressionFormat::Hevc)?;
    encoder.set_quality(HeifEncoderQuality::default())?;

    let mut heif_image = libheif_rs::HeifImage::new(
        width,
        height,
        HeifColorspace::Rgb,
        HeifChroma::InterleavedRgb,
    )?;

    // Set the RGB data
    heif_image.add_plane(
        HeifColorspace::Rgb,
        HeifChroma::InterleavedRgb,
        width,
        height,
        8,
    )?;

    let stride = heif_image.stride(0)?;
    let plane = heif_image.planes_mut().interleaved;

    // Copy pixels from the image crate buffer to the heif buffer
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let offset = (y * stride as u32 + x * 3) as usize;

            plane[offset] = pixel[0];
            plane[offset + 1] = pixel[1];
            plane[offset + 2] = pixel[2];
        }
    }

    let handle = context.encode_image(&heif_image, &encoder, None)?;
    context.set_primary_image(&handle)?;
    context.write_to_file(path)?;

    Ok(())
}

pub fn use_stirmark(
    input_dir: &Path,
    output_dir: &Path,
    stirmark_path: &Path,
) -> Result<(), Box<dyn Error>> {
    // Check if StirMark exists
    if !stirmark_path.exists() {
        return Err(format!(
            "StirMark executable not found at: {}",
            stirmark_path.display()
        )
        .into());
    }

    // Create a temporary file with the list of images to process
    let img_list_path = output_dir.join("img_list.txt");
    let mut img_list = fs::File::create(&img_list_path)?;

    // Only process JPEGs with StirMark as it has limited format support
    let entries = find_images(input_dir)?;
    let jpeg_entries: Vec<_> = entries
        .iter()
        .filter(|p| {
            if let Some(ext) = p.extension() {
                if let Some(format) = SupportedFormat::from_extension(ext.to_str().unwrap_or("")) {
                    return format == SupportedFormat::JPEG;
                }
            }
            false
        })
        .collect();

    if jpeg_entries.is_empty() {
        return Ok(());
    }

    for path in &jpeg_entries {
        use std::io::Write;
        writeln!(img_list, "{}", path.display())?;
    }

    drop(img_list);

    // Create a temporary directory for StirMark output
    let stirmark_temp_dir = output_dir.join("stirmark_temp");
    fs::create_dir_all(&stirmark_temp_dir)?;

    // Run StirMark with common transformations
    Command::new(stirmark_path)
        .args(&[
            "-i",
            img_list_path.to_str().unwrap(),
            "-o",
            stirmark_temp_dir.to_str().unwrap(),
            "-AFFINE",   // Affine transformations
            "-CONV",     // Convolution filters
            "-JPEG",     // JPEG compression
            "-NOISE",    // Add noise
            "-ROTCROP",  // Rotation with cropping
            "-ROTSCALE", // Rotation with scaling
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    // Move files to appropriate subfolders
    for entry in fs::read_dir(&stirmark_temp_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip if not a file
        if !path.is_file() {
            continue;
        }

        let filename = path.file_name().unwrap().to_str().unwrap();

        // StirMark output files are typically named with the format:
        // original_TRANSFORM.jpg
        // We need to extract the original name
        if let Some(original_name_end) = filename.find('_') {
            let original_name = &filename[0..original_name_end];

            // Create subfolder if it doesn't exist
            let subfolder = output_dir.join(original_name);
            fs::create_dir_all(&subfolder)?;

            // Move the file to the subfolder
            let target_path = subfolder.join(filename);
            fs::rename(&path, &target_path)?;
        } else {
            // If the filename doesn't follow the expected pattern,
            // just leave it in the output directory
            let target_path = output_dir.join(filename);
            fs::rename(&path, &target_path)?;
        }
    }

    // Clean up
    fs::remove_file(img_list_path)?;
    fs::remove_dir_all(stirmark_temp_dir)?;

    Ok(())
}
