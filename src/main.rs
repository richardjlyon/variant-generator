use clap::{Arg, Command};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rand::Rng;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};

// Define supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
enum SupportedFormat {
    JPEG,
    PNG,
    TIFF,
    HEIC,
}

impl SupportedFormat {
    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => Some(SupportedFormat::JPEG),
            "png" => Some(SupportedFormat::PNG),
            "tif" | "tiff" => Some(SupportedFormat::TIFF),
            "heic" => Some(SupportedFormat::HEIC),
            _ => None,
        }
    }

    fn to_image_format(&self) -> Option<ImageFormat> {
        match self {
            SupportedFormat::JPEG => Some(ImageFormat::Jpeg),
            SupportedFormat::PNG => Some(ImageFormat::Png),
            SupportedFormat::TIFF => Some(ImageFormat::Tiff),
            SupportedFormat::HEIC => None, // Special handling required
        }
    }
}

// Define transformation types
#[derive(Debug, Clone)]
enum Transformation {
    Resize(u32, u32),
    Rotate(f32),
    Flip(bool), // true = horizontal, false = vertical
    Crop(u32, u32, u32, u32),
    Blur(f32),
    Noise(f32),
    Compression(u8),
    Brightness(f32),
    Contrast(f32),
}

struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    transformations: Vec<Transformation>,
    recursive: bool,
    stirmark_path: Option<PathBuf>,
}

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
    let stirmark_path = matches
        .get_one::<String>("stirmark")
        .map(|s| PathBuf::from(s));

    let input_dir = PathBuf::from("/Users/richardlyon/Desktop/test-images/original_images");
    let output_dir = PathBuf::from("/Users/richardlyon/Desktop/test-images/output_images");

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Set up default transformations or read from config file
    let transformations = if let Some(config_path) = matches.get_one::<String>("config") {
        parse_config_file(config_path)?
    } else {
        default_transformations()
    };

    let config = Config {
        input_dir,
        output_dir,
        transformations,
        recursive,
        stirmark_path,
    };

    // Process the images
    process_images(config)?;

    Ok(())
}

fn parse_config_file(path: &str) -> Result<Vec<Transformation>, Box<dyn Error>> {
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

fn default_transformations() -> Vec<Transformation> {
    vec![
        // Resizing
        Transformation::Resize(800, 600),
        Transformation::Resize(400, 300),
        Transformation::Resize(1200, 900),
        // Rotations
        Transformation::Rotate(90.0),
        Transformation::Rotate(180.0),
        Transformation::Rotate(270.0),
        Transformation::Rotate(5.0),
        Transformation::Rotate(-5.0),
        // Flips
        Transformation::Flip(true),  // Horizontal
        Transformation::Flip(false), // Vertical
        // Cropping (percentage-based, adjusted at runtime)
        Transformation::Crop(0, 0, 0, 0), // Placeholder, will adjust based on image size
        // Blur
        Transformation::Blur(1.0),
        Transformation::Blur(3.0),
        // Noise
        Transformation::Noise(0.1),
        Transformation::Noise(0.2),
        // JPEG Compression quality
        Transformation::Compression(90),
        Transformation::Compression(50),
        Transformation::Compression(20),
        // Brightness/Contrast
        Transformation::Brightness(1.2),
        Transformation::Brightness(0.8),
        Transformation::Contrast(1.2),
        Transformation::Contrast(0.8),
    ]
}

fn process_images(config: Config) -> Result<(), Box<dyn Error>> {
    let entries = if config.recursive {
        find_images_recursive(&config.input_dir)?
    } else {
        find_images(&config.input_dir)?
    };

    for entry in entries {
        process_image(&entry, &config)?;
    }

    // If StirMark is available, use it for additional transformations
    if let Some(stirmark_path) = &config.stirmark_path {
        use_stirmark(&config.input_dir, &config.output_dir, stirmark_path)?;
    }

    Ok(())
}

fn find_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
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

fn find_images_recursive(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
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

fn process_image(img_path: &Path, config: &Config) -> Result<(), Box<dyn Error>> {
    let filename = img_path.file_name().unwrap().to_str().unwrap();
    let stem = Path::new(filename).file_stem().unwrap().to_str().unwrap();

    let ext = img_path.extension().unwrap().to_str().unwrap();
    let format = SupportedFormat::from_extension(ext).unwrap();

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
                println!(
                    "HEIC support not compiled in. Skipping: {}",
                    img_path.display()
                );
                return Ok(());
            }
        }
        _ => image::open(img_path)?,
    };

    // Apply each transformation and save the result
    for (i, transform) in config.transformations.iter().enumerate() {
        let transformed = apply_transformation(&img, transform, &format)?;

        // Generate output filename
        let transform_name = transformation_name(transform);
        let output_filename = format!("{}_{}_{}.{}", stem, transform_name, i, ext);
        let output_path = config.output_dir.join(output_filename);

        // Save the transformed image
        save_image(&transformed, &output_path, format)?;
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

fn transformation_name(transform: &Transformation) -> String {
    match transform {
        Transformation::Resize(w, h) => format!("resize_{}x{}", w, h),
        Transformation::Rotate(angle) => format!("rotate_{}", angle),
        Transformation::Flip(horizontal) => {
            if *horizontal {
                "flip_h".to_string()
            } else {
                "flip_v".to_string()
            }
        }
        Transformation::Crop(_, _, _, _) => "crop".to_string(),
        Transformation::Blur(sigma) => format!("blur_{}", sigma),
        Transformation::Noise(amount) => format!("noise_{}", amount),
        Transformation::Compression(quality) => format!("compress_{}", quality),
        Transformation::Brightness(factor) => format!("bright_{}", factor),
        Transformation::Contrast(factor) => format!("contrast_{}", factor),
    }
}

fn apply_transformation(
    img: &DynamicImage,
    transform: &Transformation,
    format: &SupportedFormat,
) -> Result<DynamicImage, Box<dyn Error>> {
    match transform {
        Transformation::Resize(width, height) => {
            Ok(img.resize(*width, *height, image::imageops::FilterType::Lanczos3))
        }
        Transformation::Rotate(angle) => {
            // Convert to RGBA8 for rotation
            let rgba = img.to_rgba8();

            // Convert angle to radians and rotate about center
            let angle_radians = angle.to_radians();
            let rotated = rotate_about_center(
                &rgba,
                angle_radians,
                Interpolation::Bilinear,
                image::Rgba([255, 255, 255, 255]), // White background
            );

            let (crop_x, crop_y, crop_width, crop_height) = calculate_rotation_crop(
                img.width(),
                img.height(),
                angle_radians,
                rotated.width(),
                rotated.height(),
            );

            // Step 3: Crop the rotated image to the tight bounding box
            // Convert rotated ImageBuffer to DynamicImage
            let rotated_img = DynamicImage::ImageRgba8(rotated);

            // Apply crop transformation
            let cropped = apply_transformation(
                &rotated_img,
                &Transformation::Crop(crop_x, crop_y, crop_width, crop_height),
                format,
            )?;

            Ok(cropped)
        }
        Transformation::Flip(horizontal) => {
            if *horizontal {
                Ok(img.fliph())
            } else {
                Ok(img.flipv())
            }
        }
        Transformation::Crop(x, y, width, height) => {
            // If the crop parameters are all 0, use percentage-based cropping
            if *x == 0 && *y == 0 && *width == 0 && *height == 0 {
                let img_width = img.width();
                let img_height = img.height();

                // Crop 20% from each side (60% of original size)
                let new_width = (img_width as f32 * 0.6) as u32;
                let new_height = (img_height as f32 * 0.6) as u32;
                let new_x = (img_width as f32 * 0.2) as u32;
                let new_y = (img_height as f32 * 0.2) as u32;

                Ok(img.crop_imm(new_x, new_y, new_width, new_height))
            } else {
                Ok(img.crop_imm(*x, *y, *width, *height))
            }
        }
        Transformation::Blur(sigma) => Ok(img.blur(*sigma)),
        Transformation::Noise(amount) => {
            // Implement simple noise addition
            let mut img = img.clone();
            add_noise(&mut img, *amount)?;
            Ok(img)
        }
        Transformation::Compression(quality) => {
            match format {
                SupportedFormat::JPEG => {
                    // For JPEG, we need to save and reload with a certain quality
                    let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                    img.write_to(&mut cursor, ImageFormat::Jpeg)?;
                    Ok(image::load_from_memory(cursor.get_ref())?)
                }
                _ => {
                    // For non-JPEG, just return the original (compression will be applied at save time)
                    Ok(img.clone())
                }
            }
        }
        Transformation::Brightness(factor) => {
            // Custom brightness implementation
            let mut img = img.clone();
            adjust_brightness(&mut img, *factor)?;
            Ok(img)
        }
        Transformation::Contrast(factor) => {
            // Custom contrast implementation
            let mut img = img.clone();
            adjust_contrast(&mut img, *factor)?;
            Ok(img)
        }
    }
}

fn add_noise(img: &mut DynamicImage, amount: f32) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();

    // Convert to RGBA8 for easier pixel manipulation
    let rgba = img.to_rgba8();
    let mut buffer = rgba.clone();

    // Get mutable access to pixels
    let (width, height) = img.dimensions();
    for y in 0..height {
        for x in 0..width {
            let mut pixel = rgba.get_pixel(x, y).clone();

            // Add random noise to each channel
            for p in pixel.0.iter_mut().take(3) {
                // Only RGB, not alpha
                let noise = (rng.gen::<f32>() - 0.5) * amount * 255.0;
                *p = (*p as f32 + noise).clamp(0.0, 255.0) as u8;
            }

            buffer.put_pixel(x, y, pixel);
        }
    }

    // Update the image
    *img = DynamicImage::ImageRgba8(buffer);

    Ok(())
}

fn adjust_brightness(img: &mut DynamicImage, factor: f32) -> Result<(), Box<dyn Error>> {
    // Convert to RGBA8 for easier pixel manipulation
    let rgba = img.to_rgba8();
    let mut buffer = rgba.clone();

    let (width, height) = img.dimensions();
    for y in 0..height {
        for x in 0..width {
            let mut pixel = rgba.get_pixel(x, y).clone();

            // Adjust brightness of RGB channels
            for p in pixel.0.iter_mut().take(3) {
                // Skip alpha
                *p = ((*p as f32) * factor).clamp(0.0, 255.0) as u8;
            }

            buffer.put_pixel(x, y, pixel);
        }
    }

    // Update the image
    *img = DynamicImage::ImageRgba8(buffer);

    Ok(())
}

fn adjust_contrast(img: &mut DynamicImage, factor: f32) -> Result<(), Box<dyn Error>> {
    // Convert to RGBA8 for easier pixel manipulation
    let rgba = img.to_rgba8();
    let mut buffer = rgba.clone();

    // Calculate the average luminance
    let (width, height) = img.dimensions();
    let mut avg_luminance = 0.0;
    let pixel_count = width * height;

    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            let luminance =
                0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
            avg_luminance += luminance / (pixel_count as f32);
        }
    }

    // Apply contrast adjustment
    for y in 0..height {
        for x in 0..width {
            let mut pixel = rgba.get_pixel(x, y).clone();

            for i in 0..3 {
                // RGB channels
                let value = pixel[i] as f32;
                let new_value = avg_luminance + factor * (value - avg_luminance);
                pixel[i] = new_value.clamp(0.0, 255.0) as u8;
            }

            buffer.put_pixel(x, y, pixel);
        }
    }

    // Update the image
    *img = DynamicImage::ImageRgba8(buffer);

    Ok(())
}

fn save_image(
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

fn use_stirmark(
    input_dir: &Path,
    output_dir: &Path,
    stirmark_path: &Path,
) -> Result<(), Box<dyn Error>> {
    println!("Running StirMark for additional transformations...");

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
    let mut img_list = File::create(&img_list_path)?;

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
        writeln!(img_list, "{}", path.display())?;
    }

    drop(img_list);

    // Run StirMark with common transformations
    let output = ProcessCommand::new(stirmark_path)
        .args(&[
            "-i",
            img_list_path.to_str().unwrap(),
            "-o",
            output_dir.to_str().unwrap(),
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

    println!("StirMark output:");
    println!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.status.success() {
        println!("StirMark error:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }

    // Clean up
    fs::remove_file(img_list_path)?;

    Ok(())
}

/// Calculate the dimensions of an image after rotation
fn calculate_rotated_dimensions(width: u32, height: u32, angle: f32) -> (u32, u32) {
    // Convert to f32 for the calculations
    let width = width as f32;
    let height = height as f32;

    // Calculate the rotated dimensions using the rotation formula
    let angle_abs = angle.abs() % (2.0 * PI);

    // Handle quadrant wrap-around
    let angle_normalized = if angle_abs > PI / 2.0 && angle_abs <= PI {
        PI - angle_abs
    } else if angle_abs > PI && angle_abs <= 3.0 * PI / 2.0 {
        angle_abs - PI
    } else if angle_abs > 3.0 * PI / 2.0 {
        2.0 * PI - angle_abs
    } else {
        angle_abs
    };

    let sin = angle_normalized.sin();
    let cos = angle_normalized.cos();

    // The dimensions of the rotated image
    let new_width = (width * cos.abs() + height * sin.abs()).ceil() as u32;
    let new_height = (width * sin.abs() + height * cos.abs()).ceil() as u32;

    (new_width, new_height)
}

/// Calculate the crop parameters for a rotated image to remove white space
fn calculate_rotation_crop(
    orig_width: u32,
    orig_height: u32,
    angle: f32,
    rotated_width: u32,
    rotated_height: u32,
) -> (u32, u32, u32, u32) {
    // Convert to signed integers for the calculations
    let width = orig_width as f32;
    let height = orig_height as f32;

    // Calculate the center point of the original image
    let center_x = width / 2.0;
    let center_y = height / 2.0;

    // Calculate the corners of the original image relative to the center
    let corners = [
        (-center_x, -center_y), // Top-left
        (center_x, -center_y),  // Top-right
        (center_x, center_y),   // Bottom-right
        (-center_x, center_y),  // Bottom-left
    ];

    // Rotate the corners
    let cos = angle.cos();
    let sin = angle.sin();
    let rotated_corners: Vec<(f32, f32)> = corners
        .iter()
        .map(|(x, y)| {
            let rx = x * cos - y * sin;
            let ry = x * sin + y * cos;
            (rx, ry)
        })
        .collect();

    // Find the min and max coordinates of the rotated corners
    let min_x = rotated_corners
        .iter()
        .map(|(x, _)| *x)
        .reduce(f32::min)
        .unwrap();
    let max_x = rotated_corners
        .iter()
        .map(|(x, _)| *x)
        .reduce(f32::max)
        .unwrap();
    let min_y = rotated_corners
        .iter()
        .map(|(_, y)| *y)
        .reduce(f32::min)
        .unwrap();
    let max_y = rotated_corners
        .iter()
        .map(|(_, y)| *y)
        .reduce(f32::max)
        .unwrap();

    // Calculate the dimensions of the tight bounding box
    let bbox_width = (max_x - min_x).ceil() as u32;
    let bbox_height = (max_y - min_y).ceil() as u32;

    // Calculate the theoretical dimensions of the fully rotated image
    let (theory_width, theory_height) =
        calculate_rotated_dimensions(orig_width, orig_height, angle);

    // The actual rotated image might have slightly different dimensions due to rounding
    // We need to adjust the crop coordinates based on the actual rotated image dimensions

    // Calculate the offset from the edge of the rotated image to the content
    let offset_x = ((rotated_width as f32 - bbox_width as f32) / 2.0).floor() as u32;
    let offset_y = ((rotated_height as f32 - bbox_height as f32) / 2.0).floor() as u32;

    // Make sure our crop stays within the bounds of the rotated image
    let crop_x = offset_x.min(rotated_width - 1);
    let crop_y = offset_y.min(rotated_height - 1);
    let crop_width = bbox_width.min(rotated_width - crop_x);
    let crop_height = bbox_height.min(rotated_height - crop_y);

    (crop_x, crop_y, crop_width, crop_height)
}
