use crate::transformations::{apply_transformation, transformation_name};
use crate::types::{Config, SupportedFormat};
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Struct to track processing statistics during image transformation
#[derive(Default, Clone)]
struct ProcessingStats {
    total_images: usize,
    successful_images: usize,
    failed_images: usize,
    total_transformations: usize,
    successful_transformations: usize,
    failed_transformations: usize,
    errors: Vec<String>,
}

/// Process images according to the provided configuration
///
/// This function is the main entry point for image processing. It:
/// 1. Finds all images in the input directory (recursively if specified)
/// 2. Processes each image with all configured transformations in parallel
/// 3. Tracks and reports progress with a progress bar
/// 4. Generates a summary report of processing results
/// 5. Optionally runs StirMark for additional transformations if configured
///
/// # Arguments
///
/// * `config` - The configuration containing input/output directories and transformation settings
///
/// # Returns
///
/// * `Result<(), Box<dyn Error>>` - Success or an error with description
pub fn process_images(config: Config) -> Result<(), Box<dyn Error>> {
    let entries = if config.recursive {
        find_images_recursive(&config.input_dir)?
    } else {
        find_images(&config.input_dir)?
    };

    if entries.is_empty() {
        println!("No images found in the input directory.");
        return Ok(());
    }

    // Calculate total number of transformations
    let total_transformations = entries.len() * config.transformations.len();
    println!(
        "Found {} images with {} transformations each ({} total steps).",
        entries.len(),
        config.transformations.len(),
        total_transformations
    );

    // Use Arc to share the Config across threads
    let config = Arc::new(config);

    // Initialize processing stats
    let stats = Arc::new(Mutex::new(ProcessingStats {
        total_images: entries.len(),
        total_transformations,
        ..Default::default()
    }));

    // Setup a single progress bar for all transformations
    let progress_bar = ProgressBar::new(total_transformations as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} transformations ({eta}) - {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress_bar.set_message("Starting...");

    let progress = Arc::new(Mutex::new(progress_bar));

    // Process images in parallel
    entries.par_iter().for_each(|entry| {
        let config = Arc::clone(&config);
        let stats = Arc::clone(&stats);
        let progress = Arc::clone(&progress);

        match process_image(entry, &config, &progress, &stats) {
            Ok(_) => {
                let mut stats = stats.lock().unwrap();
                stats.successful_images += 1;
            }
            Err(e) => {
                let mut stats = stats.lock().unwrap();
                stats.failed_images += 1;
                stats
                    .errors
                    .push(format!("Error processing {}: {}", entry.display(), e));

                // Also increment the progress bar for failed transformations
                let remaining_transformations = config.transformations.len() as u64;
                progress.lock().unwrap().inc(remaining_transformations);
            }
        }
    });

    // Ensure progress bar is finished
    progress
        .lock()
        .unwrap()
        .finish_with_message("Processing completed");

    // Print summary report
    print_summary_report(&stats.lock().unwrap());

    // If StirMark is available, use it for additional transformations
    if let Some(stirmark_path) = &config.stirmark_path {
        println!("\nRunning StirMark transformations...");
        if let Err(e) = use_stirmark(&config.input_dir, &config.output_dir, stirmark_path) {
            println!("Error running StirMark: {}", e);
        }
    }

    Ok(())
}

/// Print a summary report of the image processing results
///
/// # Arguments
///
/// * `stats` - The processing statistics to report
fn print_summary_report(stats: &ProcessingStats) {
    println!("\n===== Processing Summary =====");
    println!("Images:");
    println!("  - Total: {}", stats.total_images);
    println!("  - Successful: {}", stats.successful_images);
    println!("  - Failed: {}", stats.failed_images);

    println!("Transformations:");
    println!("  - Total: {}", stats.total_transformations);
    println!("  - Successful: {}", stats.successful_transformations);
    println!("  - Failed: {}", stats.failed_transformations);

    if !stats.errors.is_empty() {
        println!("\nErrors encountered ({}):", stats.errors.len());
        for (i, error) in stats.errors.iter().enumerate() {
            println!("  {}. {}", i + 1, error);
        }
    }
}

/// Process a single image with all configured transformations
///
/// This function:
/// 1. Creates a subfolder for the image's variants
/// 2. Loads the image (with special handling for HEIC if enabled)
/// 3. Applies each transformation and saves the result
/// 4. Updates progress and statistics
///
/// # Arguments
///
/// * `img_path` - Path to the image file
/// * `config` - Configuration with transformation settings
/// * `progress` - Progress bar for tracking and displaying progress
/// * `stats` - Statistics tracker for recording results
///
/// # Returns
///
/// * `Result<(), Box<dyn Error>>` - Success or an error with description
fn process_image(
    img_path: &Path,
    config: &Arc<Config>,
    progress: &Arc<Mutex<ProgressBar>>,
    stats: &Arc<Mutex<ProcessingStats>>,
) -> Result<(), Box<dyn Error>> {
    let filename = img_path.file_name().unwrap().to_str().unwrap();
    let stem = Path::new(filename).file_stem().unwrap().to_str().unwrap();

    let ext = img_path.extension().unwrap().to_str().unwrap();
    let format = match SupportedFormat::from_extension(ext) {
        Some(f) => f,
        None => return Err(format!("Unsupported file format: {}", ext).into()),
    };

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
                return Err("HEIC support not enabled".into());
            }
        }
        _ => match image::open(img_path) {
            Ok(img) => img,
            Err(e) => return Err(format!("Failed to open image: {}", e).into()),
        },
    };

    // Apply each transformation and save the result
    for (i, transform) in config.transformations.iter().enumerate() {
        let transform_name = transformation_name(transform);

        // Update the progress bar message
        let progress_bar = progress.lock().unwrap();
        progress_bar.set_message(format!("{} ({})", filename, transform_name));

        // Generate output filename
        let output_filename = format!("{}_{}_{}.{}", stem, transform_name, i, ext);
        let output_path = image_subfolder.join(&output_filename);

        // Skip if the file already exists (useful for resuming interrupted operations)
        if output_path.exists() && !config.force_overwrite {
            {
                let mut stats = stats.lock().unwrap();
                stats.successful_transformations += 1;
            }
            progress_bar.inc(1);
            drop(progress_bar); // Release the lock
            continue;
        }

        // Release the lock while processing
        drop(progress_bar);

        // Apply the transformation
        let transform_result = apply_transformation(&img, transform, &format);

        match transform_result {
            Ok(transformed) => {
                // Save the transformed image
                if let Err(e) = save_image(&transformed, &output_path, format) {
                    // Record the error but continue with other transformations
                    let mut stats = stats.lock().unwrap();
                    stats.failed_transformations += 1;
                    stats.errors.push(format!(
                        "Error saving transformation {} for {}: {}",
                        transform_name, filename, e
                    ));
                } else {
                    let mut stats = stats.lock().unwrap();
                    stats.successful_transformations += 1;
                }
            }
            Err(e) => {
                // Record the error but continue with other transformations
                let mut stats = stats.lock().unwrap();
                stats.failed_transformations += 1;
                stats.errors.push(format!(
                    "Error applying transformation {} to {}: {}",
                    transform_name, filename, e
                ));
            }
        }

        // Increment the progress bar
        progress.lock().unwrap().inc(1);
    }

    Ok(())
}

/// Find images in a directory (non-recursive)
///
/// # Arguments
///
/// * `dir` - Directory path to search for images
///
/// # Returns
///
/// * `Result<Vec<PathBuf>, Box<dyn Error>>` - List of image paths or an error
pub fn find_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut results = Vec::new();

    // Handle the case where the directory doesn't exist
    if !dir.exists() {
        return Err(format!("Directory not found: {}", dir.display()).into());
    }

    match fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.is_file() {
                            if let Some(ext) = path.extension() {
                                if let Some(_) =
                                    SupportedFormat::from_extension(ext.to_str().unwrap_or(""))
                                {
                                    results.push(path);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("Warning: Failed to read directory entry: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            return Err(format!("Failed to read directory {}: {}", dir.display(), e).into());
        }
    }

    Ok(results)
}

/// Find images recursively in a directory and its subdirectories
///
/// # Arguments
///
/// * `dir` - Directory path to search for images recursively
///
/// # Returns
///
/// * `Result<Vec<PathBuf>, Box<dyn Error>>` - List of image paths or an error
pub fn find_images_recursive(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut results = Vec::new();

    // Handle the case where the directory doesn't exist
    if !dir.exists() {
        return Err(format!("Directory not found: {}", dir.display()).into());
    }

    match fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.is_dir() {
                            // Continue even if a subdirectory has errors
                            match find_images_recursive(&path) {
                                Ok(subdir_images) => results.extend(subdir_images),
                                Err(e) => {
                                    println!(
                                        "Warning: Skipping subdirectory {}: {}",
                                        path.display(),
                                        e
                                    );
                                }
                            }
                        } else if path.is_file() {
                            if let Some(ext) = path.extension() {
                                if let Some(_) =
                                    SupportedFormat::from_extension(ext.to_str().unwrap_or(""))
                                {
                                    results.push(path);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("Warning: Failed to read directory entry: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            return Err(format!("Failed to read directory {}: {}", dir.display(), e).into());
        }
    }

    Ok(results)
}

/// Save an image to disk in the specified format
///
/// # Arguments
///
/// * `img` - The image to save
/// * `path` - The path where the image should be saved
/// * `format` - The format to save the image in
///
/// # Returns
///
/// * `Result<(), Box<dyn Error>>` - Success or an error with description
pub fn save_image(
    img: &DynamicImage,
    path: &Path,
    format: SupportedFormat,
) -> Result<(), Box<dyn Error>> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(
                    format!("Failed to create directory {}: {}", parent.display(), e).into(),
                );
            }
        }
    }

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
                match img.save(path.with_extension("png")) {
                    Ok(_) => (),
                    Err(e) => return Err(format!("Failed to save image as PNG: {}", e).into()),
                }
            }
        }
        _ => {
            if let Some(image_format) = format.to_image_format() {
                // Convert to RGB8 for JPEG format
                if format == SupportedFormat::JPEG {
                    let rgb8 = img.to_rgb8();
                    match DynamicImage::ImageRgb8(rgb8).save_with_format(path, image_format) {
                        Ok(_) => (),
                        Err(e) => return Err(format!("Failed to save JPEG image: {}", e).into()),
                    }
                } else {
                    match img.save_with_format(path, image_format) {
                        Ok(_) => (),
                        Err(e) => return Err(format!("Failed to save image: {}", e).into()),
                    }
                }
            } else {
                match img.save(path) {
                    Ok(_) => (),
                    Err(e) => return Err(format!("Failed to save image: {}", e).into()),
                }
            }
        }
    }

    Ok(())
}

/// Convert a HEIC image to a DynamicImage
///
/// This function is only available when the "heic" feature is enabled.
///
/// # Arguments
///
/// * `path` - Path to the HEIC image file
///
/// # Returns
///
/// * `Result<DynamicImage, Box<dyn Error>>` - The converted image or an error
#[cfg(feature = "heic")]
fn convert_heic_to_dynamic_image(path: &Path) -> Result<DynamicImage, Box<dyn Error>> {
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

    let lib_heif = LibHeif::new();
    let context = HeifContext::read_from_file(path.to_str().ok_or("Invalid path")?)?;
    let handle = context.primary_image_handle()?;
    let image = lib_heif.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)?;

    let width = image.width();
    let height = image.height();
    let planes = image.planes();
    let interleaved_plane = planes.interleaved.ok_or("No interleaved plane available")?;
    let stride = interleaved_plane.stride;
    let data = interleaved_plane.data;

    // If the stride matches what we expect for tightly packed RGB data, we can use the data directly
    if stride == width as usize * 3 {
        // Convert to RGB8 format that image crate can understand
        match image::ImageBuffer::from_raw(width as u32, height as u32, data.to_vec()) {
            Some(buffer) => Ok(DynamicImage::ImageRgb8(buffer)),
            None => {
                Err(format!(
                "Failed to create ImageBuffer from HEIC data. Dimensions: {}x{}, Data length: {}",
                width, height, data.len()
            )
                .into())
            }
        }
    } else {
        // If stride doesn't match, we need to create a new buffer with the correct stride
        // This happens when there's padding at the end of each row for memory alignment
        println!(
            "HEIC image has stride {} for width {}, repacking data",
            stride, width
        );

        // Create a new buffer with the correct stride
        let mut rgb_data = Vec::with_capacity(width as usize * height as usize * 3);

        // Copy each row, skipping the padding
        for y in 0..height as usize {
            let row_start = y * stride;
            let row_end = row_start + width as usize * 3; // 3 bytes per pixel (RGB)

            // Check if we're within bounds
            if row_end <= data.len() {
                rgb_data.extend_from_slice(&data[row_start..row_end]);
            } else {
                return Err(format!(
                    "HEIC data is truncated. Row {}: expected to read until index {} but buffer length is {}",
                    y, row_end, data.len()
                ).into());
            }
        }

        // Store the length for error reporting
        let data_len = rgb_data.len();

        // Now create the image buffer with our correctly strided data
        match image::ImageBuffer::from_raw(width as u32, height as u32, rgb_data) {
            Some(buffer) => Ok(DynamicImage::ImageRgb8(buffer)),
            None => Err(format!(
                "Failed to create ImageBuffer from repacked HEIC data. Dimensions: {}x{}, New data length: {}",
                width, height, data_len
            ).into()),
        }
    }
}

/// Save an image in HEIC format
///
/// This function is only available when the "heic" feature is enabled.
///
/// # Arguments
///
/// * `img` - The image to save
/// * `path` - The path where the image should be saved
///
/// # Returns
///
/// * `Result<(), Box<dyn Error>>` - Success or an error with description
#[cfg(feature = "heic")]
fn save_as_heic(img: &DynamicImage, path: &Path) -> Result<(), Box<dyn Error>> {
    use image::GenericImageView;
    use libheif_rs::{
        Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, Image, LibHeif,
        RgbChroma,
    };

    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let lib_heif = LibHeif::new();
    let mut context = HeifContext::new()?;
    let mut encoder = lib_heif.encoder_for_format(CompressionFormat::Hevc)?;
    encoder.set_quality(EncoderQuality::LossLess)?;

    // Create RGB image with separate planes
    let mut image = Image::new(width, height, ColorSpace::Rgb(RgbChroma::C444))?;

    // Create separate planes for R, G, B
    image.create_plane(Channel::R, width, height, 8)?;
    image.create_plane(Channel::G, width, height, 8)?;
    image.create_plane(Channel::B, width, height, 8)?;

    let planes = image.planes_mut();

    // Get mutable references to the data for each plane
    let r_plane = planes.r.ok_or("No R plane available")?;
    let g_plane = planes.g.ok_or("No G plane available")?;
    let b_plane = planes.b.ok_or("No B plane available")?;

    let r_stride = r_plane.stride;
    let g_stride = g_plane.stride;
    let b_stride = b_plane.stride;

    let r_data = r_plane.data;
    let g_data = g_plane.data;
    let b_data = b_plane.data;

    // Provide debug information
    let r_len = r_data.len();
    let g_len = g_data.len();
    let b_len = b_data.len();

    let expected_min_size = (height as usize - 1) * r_stride + width as usize;
    if r_len < expected_min_size || g_len < expected_min_size || b_len < expected_min_size {
        return Err(format!(
            "Buffer too small for image dimensions. R buffer: {}, G buffer: {}, B buffer: {}, min required: {} (width: {}, height: {})",
            r_len, g_len, b_len, expected_min_size, width, height
        ).into());
    }

    // Copy pixels from the image crate buffer to the heif buffer planes
    for y in 0..height {
        let r_row_start = y as usize * r_stride;
        let g_row_start = y as usize * g_stride;
        let b_row_start = y as usize * b_stride;

        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let x_pos = x as usize;

            // Check if we're going to access within bounds
            if r_row_start + x_pos < r_len
                && g_row_start + x_pos < g_len
                && b_row_start + x_pos < b_len
            {
                r_data[r_row_start + x_pos] = pixel[0]; // R
                g_data[g_row_start + x_pos] = pixel[1]; // G
                b_data[b_row_start + x_pos] = pixel[2]; // B
            } else {
                return Err(format!(
                    "Buffer overflow detected when writing pixel at ({}, {})",
                    x, y
                )
                .into());
            }
        }
    }

    let mut handle = context.encode_image(&image, &mut encoder, None)?;
    context.set_primary_image(&mut handle)?;
    context.write_to_file(path.to_str().ok_or("Invalid path")?)?;

    Ok(())
}

/// Use StirMark for additional image transformations
///
/// StirMark is an external tool for applying various transformations to images,
/// particularly useful for testing watermarking algorithms.
///
/// # Arguments
///
/// * `input_dir` - Directory containing input images
/// * `output_dir` - Directory where transformed images will be saved
/// * `stirmark_path` - Path to the StirMark executable
///
/// # Returns
///
/// * `Result<(), Box<dyn Error>>` - Success or an error with description
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
    let mut img_list = match fs::File::create(&img_list_path) {
        Ok(file) => file,
        Err(e) => return Err(format!("Failed to create image list file: {}", e).into()),
    };

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
        println!("No JPEG images found for StirMark processing.");
        return Ok(());
    }

    for path in &jpeg_entries {
        use std::io::Write;
        if let Err(e) = writeln!(img_list, "{}", path.display()) {
            return Err(format!("Failed to write to image list file: {}", e).into());
        }
    }

    drop(img_list);

    // Create a temporary directory for StirMark output
    let stirmark_temp_dir = output_dir.join("stirmark_temp");
    if let Err(e) = fs::create_dir_all(&stirmark_temp_dir) {
        return Err(format!("Failed to create StirMark temp directory: {}", e).into());
    }

    println!("Running StirMark on {} JPEG images...", jpeg_entries.len());

    // Run StirMark with common transformations
    let result = Command::new(stirmark_path)
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
        .output();

    match result {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("StirMark returned an error: {}", stderr);
            }
        }
        Err(e) => {
            return Err(format!("Failed to execute StirMark: {}", e).into());
        }
    }

    // Process the StirMark output files
    println!("Moving StirMark output files to appropriate directories...");
    let mut success_count = 0;
    let mut error_count = 0;

    if let Ok(entries) = fs::read_dir(&stirmark_temp_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
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
                    if let Err(e) = fs::create_dir_all(&subfolder) {
                        println!(
                            "Warning: Failed to create directory {}: {}",
                            subfolder.display(),
                            e
                        );
                        error_count += 1;
                        continue;
                    }

                    // Move the file to the subfolder
                    let target_path = subfolder.join(filename);
                    if let Err(e) = fs::rename(&path, &target_path) {
                        println!(
                            "Warning: Failed to move file to {}: {}",
                            target_path.display(),
                            e
                        );
                        error_count += 1;
                    } else {
                        success_count += 1;
                    }
                } else {
                    // If the filename doesn't follow the expected pattern,
                    // just leave it in the output directory
                    let target_path = output_dir.join(filename);
                    if let Err(e) = fs::rename(&path, &target_path) {
                        println!(
                            "Warning: Failed to move file to {}: {}",
                            target_path.display(),
                            e
                        );
                        error_count += 1;
                    } else {
                        success_count += 1;
                    }
                }
            }
        }
    }

    println!(
        "StirMark processing complete: {} files processed successfully, {} errors",
        success_count, error_count
    );

    // Clean up
    if let Err(e) = fs::remove_file(&img_list_path) {
        println!("Warning: Failed to clean up image list file: {}", e);
    }

    if let Err(e) = fs::remove_dir_all(&stirmark_temp_dir) {
        println!("Warning: Failed to clean up StirMark temp directory: {}", e);
    }

    Ok(())
}
