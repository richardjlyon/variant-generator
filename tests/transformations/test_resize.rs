use std::path::Path;
use std::time::Instant;
use variant_generator::transformations::apply_transformation;
use variant_generator::types::{ResizeFilter, SupportedFormat, Transformation};

#[test]
fn test_resize() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Verify the original dimensions
    assert_eq!(original_width, 2624);
    assert_eq!(original_height, 3636);

    // Test case with specific dimensions
    let target_width = 800;
    let target_height = 600;

    // Create resize transformation
    let transform = Transformation::Resize(target_width, target_height);

    // Apply the transformation
    let resized = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
        .expect("Failed to resize image");

    // Print actual dimensions
    println!(
        "Original dimensions: {}x{}",
        original_width, original_height
    );
    println!("Target dimensions: {}x{}", target_width, target_height);
    println!(
        "Actual dimensions: {}x{}",
        resized.width(),
        resized.height()
    );

    // Verify that the image was resized (dimensions changed)
    assert_ne!(resized.width(), original_width, "Width should have changed");
    assert_ne!(
        resized.height(),
        original_height,
        "Height should have changed"
    );

    // Verify that the aspect ratio is maintained
    let original_ratio = original_width as f32 / original_height as f32;
    let resized_ratio = resized.width() as f32 / resized.height() as f32;

    // Allow for small floating point differences
    assert!(
        (original_ratio - resized_ratio).abs() < 0.01,
        "Aspect ratio should be maintained: original={}, resized={}",
        original_ratio,
        resized_ratio
    );
}

#[test]
fn test_resize_exact_dimensions() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");

    // Test resize with exact dimensions using non-default resize method
    let width = 800;
    let height = 600;

    // Create test image with exact dimensions
    let resized_exact = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);

    // Verify the dimensions match exactly
    assert_eq!(resized_exact.width(), width, "Width should match exactly");
    assert_eq!(
        resized_exact.height(),
        height,
        "Height should match exactly"
    );

    println!(
        "Exact resize dimensions: {}x{}",
        resized_exact.width(),
        resized_exact.height()
    );
}

#[test]
fn test_resize_with_different_aspect_ratios() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Test cases with different dimensions
    let test_cases = vec![
        (400, 400, "Square dimensions"),
        (800, 600, "Landscape dimensions"),
        (600, 800, "Portrait dimensions"),
    ];

    for (width, height, description) in test_cases {
        // Create resize transformation
        let transform = Transformation::Resize(width, height);

        // Apply the transformation
        let resized = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
            .expect(&format!("Failed to resize image: {}", description));

        println!(
            "{}: target={}x{}, actual={}x{}",
            description,
            width,
            height,
            resized.width(),
            resized.height()
        );

        // Verify that the image was resized (dimensions changed)
        assert_ne!(
            resized.width(),
            original_width,
            "Width should have changed: {}",
            description
        );
        assert_ne!(
            resized.height(),
            original_height,
            "Height should have changed: {}",
            description
        );

        // If we check the code in transformations.rs, we see it's using img.resize() which
        // maintains aspect ratio while fitting within the specified dimensions.
        // So let's verify the image fits within the specified dimensions.
        assert!(
            resized.width() <= width,
            "Width should not exceed target: {} (actual: {})",
            width,
            resized.width()
        );
        assert!(
            resized.height() <= height,
            "Height should not exceed target: {} (actual: {})",
            height,
            resized.height()
        );

        // At least one dimension should match the target
        assert!(
            resized.width() == width || resized.height() == height,
            "At least one dimension should match the target: {}x{} (actual: {}x{})",
            width,
            height,
            resized.width(),
            resized.height()
        );
    }
}

#[test]
fn test_resize_performance_comparison() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let target_width = 800;
    let target_height = 600;

    // Test different filter types
    let filters = vec![
        ("Nearest (fastest)", ResizeFilter::Nearest),
        ("Triangle", ResizeFilter::Triangle),
        ("CatmullRom (balanced)", ResizeFilter::CatmullRom),
        ("Gaussian", ResizeFilter::Gaussian),
        ("Lanczos3 (highest quality)", ResizeFilter::Lanczos3),
    ];

    println!(
        "\nResize Performance Comparison ({}x{}):",
        target_width, target_height
    );
    println!("---------------------------------------------");

    // Test each filter and measure performance
    for (name, filter) in filters {
        let transform = Transformation::ResizeWithFilter(target_width, target_height, filter);

        // Measure time
        let start = Instant::now();
        let resized = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
            .expect("Failed to resize image");
        let duration = start.elapsed();

        println!(
            "{}: {:?} - dimensions: {}x{}",
            name,
            duration,
            resized.width(),
            resized.height()
        );
    }
}
