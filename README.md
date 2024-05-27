# Image Processor Application

This Image Processor Application is a simple GUI-based tool for performing various image processing operations using OpenCV and Tkinter. The application allows users to load an image, apply different filters and effects, and save the processed image. The supported operations include convolution, adding noise, applying low-pass and high-pass filters, edge detection, Gaussian filtering, blurring, mosaic effect, resizing, rotating, and a glitch effect.

## Features

- **Convolution**: Apply different convolution kernels (Laplacian, Sobel X, Sobel Y, Roberts, Prewitt) to the image.
- **Add Noise**: Add Gaussian, Salt and Pepper, Poisson, or Uniform noise to the image.
- **Low-pass Filter**: Apply a low-pass filter to smooth the image.
- **High-pass Filter**: Apply a high-pass filter to enhance edges.
- **Edge Detection**: Perform edge detection using the Canny algorithm.
- **Gaussian Filter**: Apply a Gaussian filter to blur the image.
- **Blur**: Apply a simple blur effect.
- **Mosaic**: Apply a mosaic effect to the image.
- **Resize**: Resize the image by a specified scale factor.
- **Rotate**: Rotate the image by a specified angle with support for transparent background.
- **Glitch**: Apply a glitch effect with adjustable intensity.
- **Restore**: Restore the original image.
- **Save Image**: Save the processed image.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pillow (`PIL`)
- Tkinter (included with standard Python distribution)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processor-app.git
   cd image-processor-app
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python gui.py
   ```

2. Load an image by clicking the "Choose Image" button.

3. Apply the desired image processing operations by selecting the corresponding buttons.

4. Adjust parameters for each operation through the provided dialog boxes.

5. Save the processed image by clicking the "Save Image" button.

## License

This project is open-source and available under the MIT License.

