use image::ImageFormat;
use std::path::PathBuf;

// Define supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SupportedFormat {
    JPEG,
    PNG,
    TIFF,
    HEIC,
}

impl SupportedFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => Some(SupportedFormat::JPEG),
            "png" => Some(SupportedFormat::PNG),
            "tif" | "tiff" => Some(SupportedFormat::TIFF),
            "heic" => Some(SupportedFormat::HEIC),
            _ => None,
        }
    }

    pub fn to_image_format(&self) -> Option<ImageFormat> {
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
pub enum Transformation {
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

pub struct Config {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub transformations: Vec<Transformation>,
    pub recursive: bool,
    pub stirmark_path: Option<PathBuf>,
}
