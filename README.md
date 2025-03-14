# HashTest Generator

A Rust command-line utility for generating image variants to test
perceptual hashing algorithms. This tool can process JPEG, PNG, TIFF,
and HEIC formats, applying various transformations that are useful for
evaluating the robustness of perceptual hashing algorithms.

## Features

- Processes multiple image formats: JPEG, PNG, TIFF, and HEIC (with
optional support)
- Applies common transformations including:
  - Resizing (multiple scales)
  - Rotation (various angles)
  - Flipping (horizontal and vertical)
  - Cropping
  - Blur (multiple levels)
  - Noise addition
  - JPEG compression (various quality levels)
  - Brightness adjustments
  - Contrast adjustments
- Optional StirMark integration for additional transformations
- Configurable via command-line arguments or config file

## Installation

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- For HEIC support: libheif development libraries

### Building from source

1. Clone the repository:    ```    git clone
https://github.com/yourusername/hashtest-generator.git    cd
hashtest-generator    ```

2. Build the standard version:    ```    cargo build --release    ```

3. Build with HEIC support:    ```    cargo build --release --features
heic    ```

## Usage

Basic usage:

``` hashtest-generator --input /path/to/input/images --output
/path/to/output/variants ```

With all options:

``` hashtest-generator \
  --input /path/to/input/images \
  --output /path/to/output/variants \
  --recursive \
  --stirmark /path/to/stirmark/executable \
  --config /path/to/custom/config.txt ```

### Command-line options

- `-i, --input <INPUT_DIR>`: Directory containing original images
- `-o, --output <OUTPUT_DIR>`: Directory where variant images will be
saved
- `-r, --recursive`: Search for images recursively in the input
directory
- `--stirmark <STIRMARK_PATH>`: Path to StirMark executable for
additional transformations
- `-c, --config <CONFIG_FILE>`: Custom configuration file for
transformations

### Configuration file format

The configuration file uses a simple format to define transformations:

``` # Resizing resize=800,600 resize=400,300

# Rotations rotate=90.0 rotate=180.0

# Flips flip_horizontal=true flip_vertical=true

# Blur blur=1.0 blur=3.0

# Compression quality (JPEG) compression=80 compression=40

# Brightness/contrast brightness=1.2 contrast=0.8 ```

## StirMark Integration

[StirMark](https://www.petitcolas.net/watermarking/stirmark/) is a
benchmark tool for digital watermarking that applies various
transformations to images. This utility can integrate with StirMark for
additional transformations beyond what's built in.

To use StirMark:

1. Download and compile StirMark from its official source
2. Provide the path to the StirMark executable using the `--stirmark`
option

## Example

Process all images in a directory with default transformations:

``` hashtest-generator --input ./original_images --output
./variant_images ```

This will create multiple variants of each image in the output
directory with names that indicate the transformation applied.

## HEIC Support

HEIC support is optional and requires the libheif library. To build
with HEIC support:

``` cargo build --release --features heic ```

If HEIC support is not compiled in, HEIC files will be skipped during
processing, or optionally saved as PNG when used as output.
