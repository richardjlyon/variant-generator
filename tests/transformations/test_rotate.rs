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
    // Test cases with original images and their rotated references
    let test_cases = vec![
        (
            "tests/test_images/jpg/IMG-2624x3636.jpg",
            "tests/test_images/jpg/IMG-2624x3636_rotate_5_4.jpg",
        ),
        (
            "tests/test_images/jpg/IMG-6377x4251.jpg",
            "tests/test_images/jpg/IMG-6377x4251_rotate_5_0.jpg",
        ),
    ];

    // Rotation angle to test
    let rotation_angle = 5.0;

    for (original_path, reference_path) in test_cases {
        println!("Testing rotation of: {}", original_path);

        // Load the original image
        let original_img = image::open(original_path).expect("Failed to open original image");
        let original_width = original_img.width();
        let original_height = original_img.height();

        println!(
            "Original image dimensions: {}x{}",
            original_width, original_height
        );

        // Load the reference rotated image
        let reference_img = image::open(reference_path).expect("Failed to open reference image");
        let reference_width = reference_img.width();
        let reference_height = reference_img.height();

        println!(
            "Reference rotated image dimensions: {}x{}",
            reference_width, reference_height
        );

        // Apply the rotation transformation
        let transformation = Transformation::Rotate(rotation_angle);
        let rotated_img =
            apply_transformation(&original_img, &transformation, &SupportedFormat::JPEG)
                .expect("Failed to apply rotation transformation");

        println!(
            "Our rotated image dimensions: {}x{}",
            rotated_img.width(),
            rotated_img.height()
        );

        // Assert that our rotated image's dimensions match the reference image
        assert_eq!(
            rotated_img.width(),
            reference_width,
            "Rotated width should match reference image width"
        );
        assert_eq!(
            rotated_img.height(),
            reference_height,
            "Rotated height should match reference image height"
        );
    }
}
