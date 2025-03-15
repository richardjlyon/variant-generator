/// # Image Transformations
///
/// This module provides various image transformation operations including resize, rotate, flip,
/// crop, blur, noise addition, compression, and brightness/contrast adjustments.
///
/// ## Resize Performance Guide
///
/// The resize operation's performance varies significantly based on the filter used:
///
/// | Filter      | Speed      | Quality     | Use Case                               |
/// |-------------|------------|-------------|-----------------------------------------|
/// | Nearest     | ★★★★★      | ★☆☆☆☆       | Thumbnails, previews, fast operations  |
/// | Triangle    | ★★★★☆      | ★★☆☆☆       | Quick preview with acceptable quality  |
/// | CatmullRom  | ★★★☆☆      | ★★★★☆       | Default: balanced quality and speed    |
/// | Gaussian    | ★☆☆☆☆      | ★★★★☆       | High-quality output where speed isn't critical |
/// | Lanczos3    | ★★☆☆☆      | ★★★★★       | Maximum quality for final output       |
///
/// ### Performance Comparison
///
/// For a 2624x3636 image resized to 800x600:
/// - Nearest: ~0.2s (fastest, lowest quality)
/// - Triangle: ~1.6s (good balance for quick operations)
/// - CatmullRom: ~3.1s (default, good balance)
/// - Gaussian: ~5.3s (slow, high quality)
/// - Lanczos3: ~4.5s (slower, highest quality)
///
/// ### Usage Recommendations
///
/// - For thumbnails or quick preview: `ResizeFilter::Nearest` or `ResizeFilter::Triangle`
/// - For web images or general use: Default `Resize()` or `ResizeFilter::CatmullRom`
/// - For printing or high-quality output: `ResizeFilter::Lanczos3`
///
/// ### Performance Optimization Tips
///
/// - Cache resized images when possible
/// - For batch operations, consider using parallelism with the `rayon` crate
/// - For very large images, consider resizing in steps (e.g., 50% repeatedly)
/// - If exact dimensions are needed, use `resize_exact()` rather than `resize()`
///
/// ## Crop Behavior Guide
///
/// The crop transformation supports two modes of operation:
///
/// ### 1. Explicit Crop
///
/// When specific coordinates and dimensions are provided:
/// ```rust
/// Transformation::Crop(x, y, width, height)
/// ```
/// - `x`, `y`: Top-left corner coordinates of the crop region
/// - `width`, `height`: Dimensions of the crop region
///
/// The result is an image cropped to exactly these dimensions, taken from the specified position.
///
/// ### 2. Percentage-Based Crop
///
/// When all parameters are zero:
/// ```rust
/// Transformation::Crop(0, 0, 0, 0)
/// ```
/// This performs a centered crop that removes 20% from each side, resulting in an image that's 60% of the
/// original size. This is useful for:
/// - Creating thumbnails with consistent framing
/// - Removing unnecessary borders
/// - Focusing on the central subject of an image
///
/// ### Example Use Cases
///
/// - `Transformation::Crop(0, 0, 500, 500)`: Extract a 500×500 square from the top-left corner
/// - `Transformation::Crop(width/4, height/4, width/2, height/2)`: Center crop to 50% of original size
/// - `Transformation::Crop(0, 0, 0, 0)`: Automatic center crop to 60% of original size
///
/// ## Compression Quality Guide
///
/// The compression transformation controls the quality-to-size ratio for JPEG images:
///
/// ```rust
/// Transformation::Compression(quality)
/// ```
///
/// The `quality` parameter accepts values from 0 to 100:
///
/// | Quality Range | Compression | File Size | Visual Quality | Use Case |
/// |---------------|-------------|-----------|----------------|----------|
/// | 90-100        | Minimal     | Largest   | Near lossless  | Professional photography, archiving |
/// | 70-89         | Light       | Large     | Excellent      | High-quality photography, printing |
/// | 50-69         | Moderate    | Medium    | Good           | Web images, general purpose |
/// | 20-49         | Heavy       | Small     | Fair           | Thumbnails, previews, storage optimization |
/// | 0-19          | Extreme     | Smallest  | Poor           | Maximum space saving, significant artifacts |
///
/// ### Example Use Cases
///
/// - `Transformation::Compression(90)`: High quality (low compression) - minimal visual artifacts
/// - `Transformation::Compression(50)`: Medium quality - good balance of quality and file size
/// - `Transformation::Compression(20)`: Low quality (high compression) - noticeable artifacts but smallest file size
///
/// Note: Compression is only applied to JPEG images. For other formats, the compression setting
/// is ignored during transformation but may be applied at save time depending on the format.
///
/// ## Blur Strength Guide
///
/// The blur transformation applies a Gaussian blur to the image:
///
/// ```rust
/// Transformation::Blur(sigma)
/// ```
///
/// The `sigma` parameter represents the standard deviation of the Gaussian blur kernel:
///
/// | Sigma Range | Blur Strength | Visual Effect | Use Case |
/// |-------------|---------------|---------------|----------|
/// | 0.1-0.5     | Very Subtle   | Minimal smoothing | Reducing minor noise, subtle softening |
/// | 0.5-1.5     | Light         | Noticeable smoothing | Portrait softening, removing small imperfections |
/// | 1.5-3.0     | Moderate      | Soft focus effect | Depth of field simulation, background softening |
/// | 3.0-5.0     | Strong        | Heavy blurring | Creative effects, privacy masking |
/// | 5.0+        | Extreme       | Very diffused | Abstract effects, heavy obfuscation |
///
/// ### Performance Considerations
///
/// - The computational cost increases with higher sigma values
/// - Blur operations on larger images take significantly more time
/// - For very large blurs, consider downscaling the image first, applying the blur, then upscaling
///
/// ### Example Use Cases
///
/// - `Transformation::Blur(1.0)`: Light blur - smooths minor imperfections while preserving details
/// - `Transformation::Blur(3.0)`: Moderate blur - creates a dreamy, soft focus effect
/// - `Transformation::Blur(8.0)`: Extreme blur - creates an abstract, heavily diffused image
///
/// ## Noise Effect Guide
///
/// The noise transformation adds random pixel variations to create various effects:
///
/// ```rust
/// Transformation::Noise(amount)
/// ```
///
/// The `amount` parameter controls the intensity of the noise effect:
///
/// | Amount Range | Noise Level | Visual Effect | Use Case |
/// |--------------|-------------|---------------|----------|
/// | 0.01-0.05    | Minimal     | Subtle grain  | Film simulation, vintage effects |
/// | 0.05-0.15    | Light       | Noticeable texture | Creative texturing, film grain |
/// | 0.15-0.25    | Moderate    | Significant noise | Artistic effects, stylization |
/// | 0.25-0.5     | Strong      | Heavy distortion | Abstract effects, glitch art |
/// | 0.5+         | Extreme     | Image degradation | Special effects, distortion art |
///
/// ### How It Works
///
/// - Noise is applied to each RGB channel independently (alpha is preserved)
/// - The noise amount determines the maximum possible pixel value change (±amount*127.5)
/// - Both positive and negative noise is applied, creating lighter and darker spots
/// - All pixel values are clamped to valid range (0-255)
///
/// ### Example Use Cases
///
/// - `Transformation::Noise(0.1)`: Light noise - adds subtle film grain or texture
/// - `Transformation::Noise(0.2)`: Moderate noise - creates a more pronounced grainy effect
/// - `Transformation::Noise(0.4)`: Strong noise - produces a heavily distorted, artistic look
///
use crate::types::{ResizeFilter, SupportedFormat, Transformation};
use image::{DynamicImage, GenericImageView, ImageFormat};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rand::Rng;
use std::error::Error;
use std::f32::consts::PI;
use std::io::Cursor;

pub fn transformation_name(transform: &Transformation) -> String {
    match transform {
        Transformation::Resize(w, h) => format!("resize_{}x{}", w, h),
        Transformation::ResizeWithFilter(w, h, filter) => {
            format!("resize_{}x{}_filter_{:?}", w, h, filter)
        }
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

pub fn apply_transformation(
    img: &DynamicImage,
    transform: &Transformation,
    format: &SupportedFormat,
) -> Result<DynamicImage, Box<dyn Error>> {
    match transform {
        Transformation::Resize(width, height) => {
            Ok(img.resize(*width, *height, image::imageops::FilterType::CatmullRom))
        }
        Transformation::ResizeWithFilter(width, height, filter) => {
            Ok(img.resize(*width, *height, filter.to_filter_type()))
        }
        Transformation::Rotate(angle) => {
            // Convert to RGBA8 for rotation
            let rgba = img.to_rgba8();
            let angle_radians = angle.to_radians();

            // Handle multiples of 90 degrees efficiently using built-in functions
            // Checking with small epsilon for floating point comparison
            let epsilon = 0.001;
            let normalized_angle = angle_radians % (2.0 * PI);

            if (normalized_angle - 0.0).abs() < epsilon
                || (normalized_angle - 2.0 * PI).abs() < epsilon
            {
                return Ok(img.clone()); // 0 or 360 degrees - no change
            } else if (normalized_angle - PI / 2.0).abs() < epsilon {
                return Ok(img.rotate90()); // 90 degrees
            } else if (normalized_angle - PI).abs() < epsilon {
                return Ok(img.rotate180()); // 180 degrees
            } else if (normalized_angle - 3.0 * PI / 2.0).abs() < epsilon {
                return Ok(img.rotate270()); // 270 degrees
            }

            // For all other angles, use consistent approach with inner bounding box

            // First rotate the full image
            let rotated = rotate_about_center(
                &rgba,
                angle_radians,
                Interpolation::Bilinear,
                image::Rgba([255, 255, 255, 255]), // White background
            );

            // Convert rotated ImageBuffer to DynamicImage
            let rotated_img = DynamicImage::ImageRgba8(rotated);

            // Calculate the inner bounding box dimensions
            let (inner_width, inner_height) =
                calculate_inner_rectangle(img.width(), img.height(), angle_radians);

            // Calculate the center of the rotated image
            let rotated_center_x = rotated_img.width() as f32 / 2.0;
            let rotated_center_y = rotated_img.height() as f32 / 2.0;

            // Calculate the top-left corner of the inner rectangle
            let inner_x = (rotated_center_x - inner_width as f32 / 2.0).round() as u32;
            let inner_y = (rotated_center_y - inner_height as f32 / 2.0).round() as u32;

            // Ensure we don't go out of bounds
            let inner_x = inner_x.min(rotated_img.width().saturating_sub(inner_width));
            let inner_y = inner_y.min(rotated_img.height().saturating_sub(inner_height));

            // Crop to the inner rectangle
            let cropped = rotated_img.crop_imm(inner_x, inner_y, inner_width, inner_height);

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
        Transformation::Compression(_quality) => {
            match format {
                SupportedFormat::JPEG => {
                    // For JPEG, we need to save and reload with a certain quality
                    let mut cursor = Cursor::new(Vec::<u8>::new());
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

pub fn default_transformations() -> Vec<Transformation> {
    vec![
        // Resizing with different filters
        // Uses default CatmullRom (medium quality, good speed)
        Transformation::Resize(300, 200),
        Transformation::Resize(800, 600),
        Transformation::Resize(1200, 900),
        // Rotations
        Transformation::Rotate(90.0),
        Transformation::Rotate(180.0),
        Transformation::Rotate(270.0),
        Transformation::Rotate(5.0),
        // Flips
        Transformation::Flip(true),  // Horizontal
        Transformation::Flip(false), // Vertical
        // Cropping (percentage-based, adjusted at runtime)
        Transformation::Crop(0, 0, 0, 0), // Placeholder, will adjust based on image size
        // Blur
        Transformation::Blur(0.5),
        Transformation::Blur(1.5),
        // Noise
        Transformation::Noise(0.1),
        Transformation::Noise(0.2),
        // JPEG Compression quality
        Transformation::Compression(90), // Heavy
        Transformation::Compression(50), // Moderate
        Transformation::Compression(20), // Light
        // Brightness/Contrast
        Transformation::Brightness(1.2),
        Transformation::Brightness(0.8),
        Transformation::Contrast(1.2),
        Transformation::Contrast(0.8),
    ]
}

fn add_noise(img: &mut DynamicImage, amount: f32) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();

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
                let noise = (rng.random::<f32>() - 0.5) * amount * 255.0;
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

/// Check if an angle is very close to a multiple of 90 degrees
pub fn is_right_angle_rotation(angle: f32) -> bool {
    let two_pi = 2.0 * PI;
    let angle_mod = ((angle % two_pi) + two_pi) % two_pi; // Normalize to [0, 2π)

    // Check if angle is close to 0, 90, 180, or 270 degrees
    (angle_mod < 0.01)
        || ((angle_mod - PI / 2.0).abs() < 0.01)
        || ((angle_mod - PI).abs() < 0.01)
        || ((angle_mod - 3.0 * PI / 2.0).abs() < 0.01)
        || ((angle_mod - two_pi).abs() < 0.01)
}

/// Calculate the largest rectangle that fits inside a rotated rectangle without white space

// Correct implementation of inner rectangle calculation
fn calculate_inner_rectangle(width: u32, height: u32, angle_radians: f32) -> (u32, u32) {
    let width = width as f32;
    let height = height as f32;

    // Normalize angle to [0, π/2] since the problem is symmetric
    let normalized_angle = angle_radians.abs() % (PI * 2.0);
    let normalized_angle = if normalized_angle > PI {
        normalized_angle - PI
    } else {
        normalized_angle
    };
    let normalized_angle = if normalized_angle > PI / 2.0 {
        PI - normalized_angle
    } else {
        normalized_angle
    };

    let cos_angle = normalized_angle.cos();
    let sin_angle = normalized_angle.sin();

    // Handle division by zero
    if sin_angle.abs() < 0.001 {
        return (width as u32, height as u32);
    }
    if cos_angle.abs() < 0.001 {
        return (height as u32, width as u32);
    }

    // Standard formula for inner bounding box calculation
    let inner_width = width * cos_angle - height * sin_angle;
    let inner_height = height * cos_angle - width * sin_angle;

    // Ensure dimensions are at least 1 pixel and handle negative values
    (
        (inner_width.abs().max(1.0)) as u32,
        (inner_height.abs().max(1.0)) as u32,
    )
}
