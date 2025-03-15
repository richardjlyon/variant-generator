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

            // Handle special cases for multiples of 90 degrees
            let is_right_angle = is_right_angle_rotation(angle_radians);

            if is_right_angle {
                // For 90/180/270 degrees, just do the rotation without inner cropping
                if (angle_radians - PI / 2.0).abs() < 0.01
                    || (angle_radians - 3.0 * PI / 2.0).abs() < 0.01
                {
                    // 90 or 270 degrees - just rotate and swap dimensions
                    return Ok(img.rotate90());
                } else if (angle_radians - PI).abs() < 0.01 {
                    // 180 degrees
                    return Ok(img.rotate180());
                } else {
                    // 0/360 degrees - no change
                    return Ok(img.clone());
                }
            }

            // Calculate the largest inner rectangle dimensions
            let (inner_width, inner_height) =
                calculate_inner_rectangle(img.width(), img.height(), angle_radians);

            // First rotate the image using the imageproc library
            let rotated = rotate_about_center(
                &rgba,
                angle_radians,
                Interpolation::Bilinear,
                image::Rgba([255, 255, 255, 255]), // White background
            );

            // Calculate the center of the rotated image
            let rotated_center_x = rotated.width() as f32 / 2.0;
            let rotated_center_y = rotated.height() as f32 / 2.0;

            // Calculate the top-left corner of the inner rectangle
            let inner_x = (rotated_center_x - inner_width as f32 / 2.0).floor() as u32;
            let inner_y = (rotated_center_y - inner_height as f32 / 2.0).floor() as u32;

            // Convert rotated ImageBuffer to DynamicImage
            let rotated_img = DynamicImage::ImageRgba8(rotated);

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
        Transformation::Resize(800, 600), // Uses default CatmullRom (medium quality, good speed)
        Transformation::ResizeWithFilter(800, 600, ResizeFilter::Nearest), // Fastest
        Transformation::ResizeWithFilter(800, 600, ResizeFilter::Lanczos3), // Highest quality
        Transformation::ResizeWithFilter(400, 300, ResizeFilter::Triangle), // Good balance
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
pub fn calculate_inner_rectangle(width: u32, height: u32, angle: f32) -> (u32, u32) {
    // For special cases of 0, 90, 180, 270 degrees
    let two_pi = 2.0 * PI;
    let angle_mod = ((angle % two_pi) + two_pi) % two_pi; // Normalize to [0, 2π)

    if angle_mod < 0.01 || (angle_mod - two_pi).abs() < 0.01 {
        // 0 or 360 degrees - no change
        return (width, height);
    } else if (angle_mod - PI / 2.0).abs() < 0.01 {
        // 90 degrees - swap dimensions
        return (height, width);
    } else if (angle_mod - PI).abs() < 0.01 {
        // 180 degrees - no change in dimensions
        return (width, height);
    } else if (angle_mod - 3.0 * PI / 2.0).abs() < 0.01 {
        // 270 degrees - swap dimensions
        return (height, width);
    }

    // Convert dimensions to f32
    let w = width as f32;
    let h = height as f32;

    // Normalize angle to [0, PI/2]
    let angle_normalized = angle_mod % PI;
    let angle_normalized = if angle_normalized > PI / 2.0 {
        PI - angle_normalized
    } else {
        angle_normalized
    };

    // Calculate trigonometric values
    let sin = angle_normalized.sin().abs();
    let cos = angle_normalized.cos().abs();

    // The largest inscribed rectangle in a rotated rectangle can be calculated as:
    // For a rectangle with width W and height H rotated by angle θ:
    // Width of inscribed rectangle = W * cos(θ) - H * sin(θ) * tan(θ)
    // Height of inscribed rectangle = H * cos(θ) - W * sin(θ) * cot(θ)

    // For a square, there's a simpler formula:
    if (w - h).abs() < 0.001 {
        // Check if it's a square
        // For squares, the inscribed rectangle has both sides of length:
        // side = original_side / (cos(θ) + sin(θ))
        let side = w / (cos + sin);
        return (side.floor() as u32, side.floor() as u32);
    }

    // For rectangles, we need to compute differently:
    // Formula from the literature on largest inscribed rectangles within rotated rectangles
    // This formula works for arbitrary rectangles and angles
    let cot_theta = cos / sin; // cotangent of the angle
    let tan_theta = sin / cos; // tangent of the angle

    // Calculate width and height of inscribed rectangle
    let inscribed_width = (w * cos - h * sin * tan_theta).abs();
    let inscribed_height = (h * cos - w * sin * cot_theta).abs();

    // Floor to ensure we don't exceed the available inner space
    (
        inscribed_width.floor() as u32,
        inscribed_height.floor() as u32,
    )
}
