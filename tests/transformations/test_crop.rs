use std::path::Path;
use variant_generator::transformations::apply_transformation;
use variant_generator::types::{SupportedFormat, Transformation};

#[test]
fn test_explicit_crop() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Verify the original dimensions
    assert_eq!(original_width, 2624);
    assert_eq!(original_height, 3636);

    // Test case with explicit crop coordinates
    let x = 500;
    let y = 800;
    let crop_width = 1000;
    let crop_height = 1500;

    // Create crop transformation
    let transform = Transformation::Crop(x, y, crop_width, crop_height);

    // Apply the transformation
    let cropped = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
        .expect("Failed to crop image");

    // Print dimensions
    println!(
        "Original dimensions: {}x{}",
        original_width, original_height
    );
    println!(
        "Crop parameters: x={}, y={}, width={}, height={}",
        x, y, crop_width, crop_height
    );
    println!(
        "Cropped dimensions: {}x{}",
        cropped.width(),
        cropped.height()
    );

    // Verify that the image was cropped to the correct dimensions
    assert_eq!(cropped.width(), crop_width, "Width should match crop width");
    assert_eq!(
        cropped.height(),
        crop_height,
        "Height should match crop height"
    );
}

#[test]
fn test_percentage_crop() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Create percentage-based crop transformation (all zeros means a percentage-based crop)
    let transform = Transformation::Crop(0, 0, 0, 0);

    // Apply the transformation
    let cropped = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
        .expect("Failed to crop image");

    // Calculate expected dimensions (60% of original size from the transformations.rs implementation)
    let expected_width = (original_width as f32 * 0.6) as u32;
    let expected_height = (original_height as f32 * 0.6) as u32;

    // Print dimensions
    println!(
        "Original dimensions: {}x{}",
        original_width, original_height
    );
    println!(
        "Expected cropped dimensions (60% of original): {}x{}",
        expected_width, expected_height
    );
    println!(
        "Actual cropped dimensions: {}x{}",
        cropped.width(),
        cropped.height()
    );

    // Verify that the image was cropped to the expected dimensions
    assert_eq!(
        cropped.width(),
        expected_width,
        "Width should be 60% of original"
    );
    assert_eq!(
        cropped.height(),
        expected_height,
        "Height should be 60% of original"
    );
}

#[test]
fn test_crop_bounds() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Test cases for different crop bounds
    let test_cases = vec![
        // Edge crop - top-left corner
        (0, 0, 500, 500, "Top-left corner"),
        // Edge crop - bottom-right corner
        (
            original_width - 500,
            original_height - 500,
            500,
            500,
            "Bottom-right corner",
        ),
        // Center crop
        (
            original_width / 4,
            original_height / 4,
            original_width / 2,
            original_height / 2,
            "Center crop",
        ),
        // Letterbox-like crop (keep width, reduce height)
        (
            0,
            original_height / 4,
            original_width,
            original_height / 2,
            "Letterbox crop",
        ),
    ];

    for (x, y, width, height, description) in test_cases {
        // Create crop transformation
        let transform = Transformation::Crop(x, y, width, height);

        // Apply the transformation
        let cropped = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
            .expect(&format!("Failed to crop image: {}", description));

        println!(
            "{}: x={}, y={}, target={}x{}, actual={}x{}",
            description,
            x,
            y,
            width,
            height,
            cropped.width(),
            cropped.height()
        );

        // Verify that the image was cropped to the expected dimensions
        assert_eq!(
            cropped.width(),
            width,
            "Width should match target: {}",
            description
        );
        assert_eq!(
            cropped.height(),
            height,
            "Height should match target: {}",
            description
        );
    }
}
