# Variant Generator

A Rust command-line utility for generating image variants to test perceptual hashing algorithms and image processing systems. This tool processes JPEG, PNG, TIFF, and optionally HEIC formats, applying various transformations that are useful for evaluating the robustness of perceptual hashing algorithms or generating training data for computer vision models.

## Features

- **Multi-format Support**: Process JPEG, PNG, TIFF, and HEIC images (HEIC requires optional compilation with the `heic` feature)
- **Parallel Processing**: Uses Rayon for efficient multi-threaded processing of images
- **Progress Tracking**: Real-time progress bar with ETA and completion status
- **Resume Capability**: Can skip already processed images when restarting an interrupted job
- **Comprehensive Transformations**:
  - **Resizing**: Scale images to various dimensions
  - **Rotation**: Rotate by any angle with intelligent boundary handling
  - **Flipping**: Horizontal and vertical mirroring
  - **Cropping**: Extract regions of interest
  - **Blur**: Apply Gaussian blur at different intensities
  - **Noise Addition**: Add random noise at configurable levels
  - **JPEG Compression**: Test resilience to compression artifacts
  - **Brightness Adjustment**: Increase or decrease brightness
  - **Contrast Adjustment**: Modify image contrast
- **StirMark Integration**: Optional support for [StirMark](https://www.petitcolas.net/watermarking/stirmark/) to apply additional complex transformations
- **Configurable**: Use command-line arguments or a configuration file to customize transformations

## Installation

### Prerequisites

- **Rust Toolchain**: Install via [rustup](https://rustup.rs/)
- **For HEIC Support**: libheif development libraries (optional)
  - On macOS: `brew install libheif`
  - On Ubuntu/Debian: `apt install libheif-dev`
  - On Windows: Follow the [libheif documentation](https://github.com/strukturag/libheif)

### Building from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/variant-generator.git
   cd variant-generator
   ```

2. **Build standard version**:
   ```bash
   cargo build --release
   ```

3. **Build with HEIC support**:
   ```bash
   cargo build --release --features heic
   ```

4. **Install for system-wide use** (optional):
   ```bash
   cargo install --path .
   ```

## Usage

### Basic Usage

```bash
variant-generator --input /path/to/input/images --output /path/to/output/variants
```

### Advanced Usage

```bash
variant-generator \
  --input /path/to/input/images \
  --output /path/to/output/variants \
  --recursive \
  --force \
  --stirmark /path/to/stirmark/executable \
  --config /path/to/custom/config.txt
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-i, --input <DIR>` | Directory containing original images (defaults to current directory) |
| `-o, --output <DIR>` | Directory where variant images will be saved (defaults to 'variants' subfolder) |
| `-r, --recursive` | Search for images recursively in the input directory |
| `-f, --force` | Force overwrite of existing output files |
| `--stirmark <PATH>` | Path to StirMark executable for additional transformations |
| `-c, --config <FILE>` | Custom configuration file for transformations |

### Configuration File Format

The configuration file uses a simple format to define transformations, with one transformation per line:

```
# Resizing
resize=800,600
resize=400,300

# Rotations
rotate=90.0
rotate=180.0
rotate=45.0

# Flips
flip_horizontal=true
flip_vertical=true

# Cropping (x, y, width, height)
crop=10,10,500,500

# Blur
blur=1.0
blur=3.0

# Noise (0.0-1.0)
noise=0.1
noise=0.3

# Compression quality (JPEG) - value from 1-100
compression=80
compression=40

# Brightness/contrast adjustment - values above 1.0 increase, below 1.0 decrease
brightness=1.2
contrast=0.8
```

## Output Organization

The transformed images are saved with organized filenames and directory structure:

- Each original image gets its own subfolder in the output directory
- Transformed images follow the naming pattern: `[original_name]_[transformation]_[index].[extension]`
- Example: `cat_rotate_90_0.jpg`, `cat_blur_3.0_1.jpg`

## StirMark Integration

[StirMark](https://www.petitcolas.net/watermarking/stirmark/) is a benchmark tool originally designed for testing digital watermarking algorithms. This utility can integrate with StirMark for additional transformations beyond the built-in ones.

### Using StirMark

1. Download and compile StirMark from its official source
2. Provide the path to the StirMark executable using the `--stirmark` option
3. The tool will automatically process JPEG images with StirMark's transformations and organize the results

The following StirMark transformations are applied:
- Affine transformations
- Convolution filters
- JPEG compression at various quality levels
- Noise addition
- Rotation with cropping
- Rotation with scaling

## HEIC Support

HEIC (High Efficiency Image Container) support is optional and requires the libheif library. Files in HEIC format are becoming increasingly common as the default format for photos on iOS devices.

### Building with HEIC Support

```bash
cargo build --release --features heic
```

If HEIC support is not compiled in, HEIC files will be skipped during processing, or optionally saved as PNG when used as output format.

## Examples

### Process a Single Directory

```bash
variant-generator --input ./photos --output ./variants
```

### Process Images Recursively with Default Transformations

```bash
variant-generator --input ./photos --output ./variants --recursive
```

### Use a Custom Configuration File

```bash
variant-generator --input ./photos --output ./variants --config ./my_config.txt
```

### Resume a Previous Run

If a processing job was interrupted, you can resume it by running the same command again. The tool will skip already processed images:

```bash
variant-generator --input ./photos --output ./variants
```

To force reprocessing of all images, add the `--force` flag:

```bash
variant-generator --input ./photos --output ./variants --force
```

## Performance Considerations

- The tool uses parallel processing for optimal performance
- Memory usage depends on image size and number of cores used
- For large datasets, consider monitoring system resources
- SSD storage will significantly improve processing speed compared to HDD

## Troubleshooting

If you encounter any issues:

1. Ensure you have the latest version
2. Check that input images are valid and readable
3. Verify you have write permissions for the output directory
4. For HEIC issues, confirm that libheif is properly installed
5. For StirMark integration issues, verify the executable path is correct

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request
