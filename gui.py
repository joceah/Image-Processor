import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog, HORIZONTAL
from PIL import Image, ImageTk
import tkinter as tk

cover_img = cv2.imread("images/cover.jpg", cv2.IMREAD_UNCHANGED)


def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def add_noise(image, noise_type="gaussian", noise_level=10):
    if noise_level == 0:
        return image
    if noise_type == "gaussian":
        mean = 0
        sigma = noise_level ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy_image = cv2.add(image, gauss)
        return noisy_image
    elif noise_type == "salt_and_pepper":
        s_vs_p = 0.5
        amount = noise_level / 100.0
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out.astype(np.uint8)
    elif noise_type == "poisson":
        vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
        noisy = np.random.poisson(image * noise_level / 255.0 * vals) / float(vals) * 255.0 / noise_level * 60
        return noisy.astype('uint8')
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, image.shape).astype('uint8')
        noisy_image = cv2.add(image, noise)
        return noisy_image


def low_pass_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)


def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def edge_detection(image):
    return cv2.Canny(image, 100, 200)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_effects(image, effect_type="blur", mosaic_degree=10):
    if effect_type == "blur":
        return cv2.blur(image, (5, 5))
    elif effect_type == "mosaic":
        small = cv2.resize(image, (mosaic_degree, mosaic_degree), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    matrix[0, 2] += (nW / 2) - cX
    matrix[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(image, matrix, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


def apply_glitch(image, glitch_degree=5):
    rows, cols, _ = image.shape
    for i in range(0, rows, glitch_degree):
        dx = np.random.randint(-glitch_degree, glitch_degree)
        image[i:i + glitch_degree, :] = np.roll(image[i:i + glitch_degree, :], dx, axis=1)
    glitch_img = np.copy(image)
    for c in range(3):
        dx = np.random.randint(-glitch_degree, glitch_degree)
        glitch_img[:, :, c] = np.roll(glitch_img[:, :, c], dx, axis=1)
    return glitch_img


def open_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def save_file(image):
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                        ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, image)


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.state("zoom")
        self.master.title("Image Processor")
        self.master.configure(bg='lightgrey')
        top_frame = tk.Frame(master, bg='lightgrey')
        top_frame.pack(pady=10)

        self.choose_button = tk.Button(top_frame, text="Choose Image", command=self.choose_image, font=('Arial', 12),
                                       width=15, height=2)
        self.choose_button.pack(side=tk.LEFT, padx=5, pady=30)

        self.save_button = tk.Button(top_frame, text="Save Image", command=self.save_image, font=('Arial', 12),
                                     width=15, height=2)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.pack_forget()  # Hide the save button initially

        self.label = tk.Label(master, text="Select an image to process", font=('Arial', 14), bg='lightgrey')
        self.label.pack(pady=10)

        self.process_frame = tk.Frame(master, bg='lightgrey')
        self.process_frame.pack(pady=10)

        self.button_texts_commands = [
            ("Convolve", self.convolve_image_dialog),
            ("Add Noise", self.add_noise_image_dialog),
            ("Low-pass Filter", self.low_pass_filter_image),
            ("High-pass Filter", self.high_pass_filter_image),
            ("Edge Detection", self.edge_detection_image),
            ("Gaussian Filter", self.gaussian_filter_image),
            ("Blur", self.blur_image),
            ("Mosaic", self.mosaic_image_dialog),
            ("Resize", self.resize_image_dialog),
            ("Rotate", self.rotate_image_dialog),
            ("Glitch", self.glitch_image_dialog),
            ("Restore", self.restore_image)
        ]

        self.button_widgets = []
        for i, (text, command) in enumerate(self.button_texts_commands):
            button = tk.Button(self.process_frame, text=text, command=command, font=('Arial', 10), width=15, height=2)
            button.grid(row=i // 6, column=i % 6, padx=5, pady=5)
            button.grid_remove()  # Initially hide the buttons
            self.button_widgets.append(button)

        self.image_label = tk.Label(master, bg='lightgrey')
        self.image_label.pack(pady=10)
        self.display_image(cover_img)

        self.original_image = None
        self.processed_image = None
        self.preview_image = None

    def enable_buttons(self):
        for button in self.button_widgets:
            button.grid()  # Show the buttons
        self.save_button.pack(side=tk.LEFT, padx=5)  # Show the save button

    def choose_image(self):
        self.file_path = filedialog.askopenfilename()
        self.original_image = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.preview_image = self.original_image.copy()
            self.display_image(self.processed_image)
            self.enable_buttons()

    def convolve_image_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Convolve")
        dialog.grid_columnconfigure(0, weight=1)

        tk.Label(dialog, text="Choose kernel type:", font=('Arial', 10)).grid(row=0, column=0, padx=10, pady=5)

        kernel_type = tk.StringVar(dialog)
        kernel_type.set("laplacian")  # default value
        kernel_type_menu = tk.OptionMenu(dialog, kernel_type, "laplacian", "sobel_x", "sobel_y", "roberts", "prewitt")
        kernel_type_menu.grid(row=1, column=0, padx=10, pady=5)

        tk.Label(dialog, text="Convert to grayscale:", font=('Arial', 10)).grid(row=2, column=0, padx=10, pady=5)
        grayscale_var = tk.BooleanVar()
        grayscale_check = tk.Checkbutton(dialog, variable=grayscale_var)
        grayscale_check.grid(row=3, column=0, padx=10, pady=5)

        def apply_convolution_kernel():
            kernel_type_value = kernel_type.get()
            predefined_kernels = {
                "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                "roberts": np.array([[1, 0], [0, -1]]),
                "prewitt": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            }
            kernel = predefined_kernels[kernel_type_value]
            if grayscale_var.get():
                gray_image = cv2.cvtColor(self.preview_image, cv2.COLOR_BGR2GRAY)
                self.preview_image = apply_convolution(gray_image, kernel)
                self.processed_image = self.preview_image.copy()
            else:
                self.preview_image = apply_convolution(self.preview_image, kernel)
                self.processed_image = self.preview_image.copy()
            self.display_image(self.preview_image)

        def cancel_changes():
            self.preview_image = self.processed_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel_changes)

        apply_button = tk.Button(dialog, text="Apply", command=lambda: [apply_convolution_kernel(), dialog.destroy()],
                                 font=('Arial', 10), width=15, height=2)
        apply_button.grid(row=4, column=0, pady=10)

        cancel_button = tk.Button(dialog, text="Cancel", command=cancel_changes, font=('Arial', 10), width=15, height=2)
        cancel_button.grid(row=5, column=0, pady=10)

    def add_noise_image_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Add Noise")

        tk.Label(dialog, text="Choose noise type:", font=('Arial', 10)).pack(pady=5)

        noise_type = tk.StringVar(dialog)
        noise_type.set("gaussian")  # default value
        noise_type_menu = tk.OptionMenu(dialog, noise_type, "gaussian", "salt_and_pepper", "poisson", "uniform")
        noise_type_menu.pack(pady=5)

        tk.Label(dialog, text="Adjust noise level:", font=('Arial', 10)).pack(pady=5)

        noise_level_slider = tk.Scale(dialog, from_=0, to=100, orient=HORIZONTAL,
                                      command=lambda val: self.update_noise_preview(noise_type.get(),
                                                                                    noise_level_slider.get()))
        noise_level_slider.pack(pady=5)

        def apply_noise():
            self.processed_image = self.preview_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        def cancel_changes():
            self.preview_image = self.processed_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel_changes)

        apply_button = tk.Button(dialog, text="Apply", command=apply_noise, font=('Arial', 10), width=15, height=2)
        apply_button.pack(pady=10)

        cancel_button = tk.Button(dialog, text="Cancel", command=cancel_changes, font=('Arial', 10), width=15, height=2)
        cancel_button.pack(pady=10)

    def update_noise_preview(self, noise_type, noise_level):
        if self.original_image is not None:
            self.preview_image = add_noise(self.processed_image, noise_type=noise_type, noise_level=noise_level)
            self.display_image(self.preview_image)

    def low_pass_filter_image(self):
        self.preview_image = low_pass_filter(self.processed_image)
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def high_pass_filter_image(self):
        self.preview_image = high_pass_filter(self.processed_image)
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def edge_detection_image(self):
        self.preview_image = edge_detection(self.processed_image)
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def gaussian_filter_image(self):
        self.preview_image = gaussian_filter(self.processed_image)
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def blur_image(self):
        self.preview_image = apply_effects(self.processed_image, effect_type="blur")
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def mosaic_image_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Mosaic Effect")

        tk.Label(dialog, text="Adjust mosaic degree:", font=('Arial', 10)).pack(pady=5)

        mosaic_degree_slider = tk.Scale(dialog, from_=1, to=100, orient=HORIZONTAL,
                                        command=lambda val: self.update_mosaic_preview(mosaic_degree_slider.get()))
        mosaic_degree_slider.pack(pady=5)

        def apply_mosaic():
            self.processed_image = self.preview_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        def cancel_changes():
            self.preview_image = self.processed_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel_changes)

        apply_button = tk.Button(dialog, text="Apply", command=apply_mosaic, font=('Arial', 10), width=15, height=2)
        apply_button.pack(pady=10)

        cancel_button = tk.Button(dialog, text="Cancel", command=cancel_changes, font=('Arial', 10), width=15, height=2)
        cancel_button.pack(pady=10)

    def update_mosaic_preview(self, mosaic_degree):
        if self.original_image is not None:
            self.preview_image = apply_effects(self.processed_image, effect_type="mosaic",
                                               mosaic_degree=101 - mosaic_degree)
            self.display_image(self.preview_image)

    def resize_image_dialog(self):
        scale = float(simpledialog.askstring("Input", "Enter scale factor:"))
        self.preview_image = resize_image(self.processed_image, scale)
        self.display_image(self.preview_image)
        self.processed_image = self.preview_image.copy()

    def rotate_image_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Rotate Image")

        def update_rotation_text(val):
            angle_entry.delete(0, tk.END)
            angle_entry.insert(0, val)
            self.update_rotation_preview(int(val))

        tk.Label(dialog, text="Adjust rotation angle:", font=('Arial', 10)).pack(pady=5)
        angle_slider = tk.Scale(dialog, from_=0, to=360, orient=HORIZONTAL, command=update_rotation_text)
        angle_slider.pack(pady=5)

        tk.Label(dialog, text="Or enter rotation angle:", font=('Arial', 10)).pack(pady=5)
        angle_entry = tk.Entry(dialog)
        angle_entry.pack(pady=5)

        def update_rotation_slider(*args):
            try:
                angle = int(angle_entry.get())
                angle_slider.set(angle)
                self.update_rotation_preview(angle)
            except ValueError:
                pass

        angle_entry.bind("<Return>", update_rotation_slider)

        def apply_rotation():
            self.processed_image = self.preview_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        def cancel_changes():
            self.preview_image = self.processed_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel_changes)

        apply_button = tk.Button(dialog, text="Apply", command=apply_rotation, font=('Arial', 10), width=15, height=2)
        apply_button.pack(pady=10)

        cancel_button = tk.Button(dialog, text="Cancel", command=cancel_changes, font=('Arial', 10), width=15, height=2)
        cancel_button.pack(pady=10)

    def update_rotation_preview(self, angle):
        if self.original_image is not None:
            self.preview_image = rotate_image(self.processed_image, angle)
            self.display_image(self.preview_image)

    def glitch_image_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Glitch Effect")

        tk.Label(dialog, text="Adjust glitch degree:", font=('Arial', 10)).pack(pady=5)

        glitch_degree_slider = tk.Scale(dialog, from_=1, to=20, orient=HORIZONTAL,
                                        command=lambda val: self.update_glitch_preview(glitch_degree_slider.get()))
        glitch_degree_slider.pack(pady=5)

        def apply_glitch_effect():
            self.processed_image = self.preview_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        def cancel_changes():
            self.preview_image = self.processed_image.copy()
            self.display_image(self.processed_image)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel_changes)

        apply_button = tk.Button(dialog, text="Apply", command=apply_glitch_effect, font=('Arial', 10), width=15,
                                 height=2)
        apply_button.pack(pady=10)

        cancel_button = tk.Button(dialog, text="Cancel", command=cancel_changes, font=('Arial', 10), width=15, height=2)
        cancel_button.pack(pady=10)

    def update_glitch_preview(self, glitch_degree):
        if self.original_image is not None:
            self.preview_image = apply_glitch(self.processed_image.copy(), glitch_degree=glitch_degree)
            self.display_image(self.preview_image)

    def restore_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.preview_image = self.original_image.copy()
            self.display_image(self.processed_image)

    def save_image(self):
        if self.processed_image is not None:
            save_file(self.processed_image)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
