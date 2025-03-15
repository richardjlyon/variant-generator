use std::path::Path;
use variant_generator::transformations::{apply_transformation, is_right_angle_rotation};
use variant_generator::types::{SupportedFormat, Transformation};

#[test]
fn test_rotate_right_angles() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Verify the original dimensions
    assert_eq!(original_width, 2624);
    assert_eq!(original_height, 3636);

    // Test right angle rotations (90, 180, 270 degrees)
    let right_angle_tests = vec![
        (90.0, original_height, original_width, "90 degrees"),
        (180.0, original_width, original_height, "180 degrees"),
        (270.0, original_height, original_width, "270 degrees"),
        (
            0.0,
            original_width,
            original_height,
            "0 degrees/360 degrees",
        ),
    ];

    for (angle, expected_width, expected_height, description) in right_angle_tests {
        // Create rotation transformation
        let transform = Transformation::Rotate(angle);

        // Apply the transformation
        let rotated = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
            .expect(&format!("Failed to rotate image: {}", description));

        println!(
            "{}: Original dimensions: {}x{}, Rotated dimensions: {}x{}",
            description,
            original_width,
            original_height,
            rotated.width(),
            rotated.height()
        );

        // Check that the rotated image has the expected dimensions
        assert_eq!(
            rotated.width(),
            expected_width,
            "Rotated width should match expected for {}",
            description
        );
        assert_eq!(
            rotated.height(),
            expected_height,
            "Rotated height should match expected for {}",
            description
        );

        // Verify it's correctly identified as a right angle rotation
        assert!(
            is_right_angle_rotation(angle.to_radians()),
            "{} should be identified as a right angle rotation",
            description
        );
    }
}

#[test]
fn test_rotate_arbitrary_angles() {
    // Test image path
    let img_path = Path::new("tests/test_images/jpg/IMG-2624x3636.jpg");

    // Load the test image
    let img = image::open(img_path).expect("Failed to open test image");
    let original_width = img.width();
    let original_height = img.height();

    // Test non-right angle rotations
    let angle_tests = vec![
        (45.0, "45 degrees"),
        (30.0, "30 degrees"),
        (-15.0, "-15 degrees"),
        (5.0, "5 degrees"),
    ];

    for (angle, description) in angle_tests {
        // Create rotation transformation
        let transform = Transformation::Rotate(angle);

        // Apply the transformation
        let rotated = apply_transformation(&img, &transform, &SupportedFormat::JPEG)
            .expect(&format!("Failed to rotate image: {}", description));

        println!(
            "{}: Original dimensions: {}x{}, Rotated dimensions: {}x{}",
            description,
            original_width,
            original_height,
            rotated.width(),
            rotated.height()
        );

        // For arbitrary rotations, dimensions will be different but the
        // rotated image should be smaller than the original
        assert!(
            rotated.width() < original_width || rotated.height() < original_height,
            "At least one dimension should be smaller than the original for {}",
            description
        );

        // Verify it's correctly identified as NOT a right angle rotation
        assert!(
            !is_right_angle_rotation(angle.to_radians()),
            "{} should NOT be identified as a right angle rotation",
            description
        );
    }
}
