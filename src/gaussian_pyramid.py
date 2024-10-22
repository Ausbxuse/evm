import numpy as np
import tqdm
import matplotlib.pyplot as plt

from processing import idealTemporalBandpassFilter, pyrDown, pyrUp, rgb2yiq


def generateGaussianPyramid(image, kernel, level):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
            image=gaussian_pyramid,
            kernel=kernel,
            dst_shape=image_shape[level - curr_level - 1],
        )

    return gaussian_pyramid


def get_frequency_map(images, freq_range, fps=30):
    images = rgb2yiq(images)
    window_radius = 4
    center_point = (264, 120)
    print("images shape", images.shape)
    green_channel = images[
        :,
        center_point[0] - window_radius : center_point[0] + window_radius + 1,
        center_point[1] - window_radius : center_point[1] + window_radius + 1,
        1,
    ]

    intensity_over_time = green_channel.mean(axis=(1, 2))

    # Plot the intensity over time
    plt.figure(figsize=(10, 5))
    plt.plot(intensity_over_time, label="Green Channel Intensity", color="green")
    plt.xlabel("Frame")
    plt.ylabel("Average Intensity")
    plt.title("Green Channel Intensity Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("intensity.png")
    green_channel = green_channel.astype(np.float32) / 255.0
    fft = np.fft.fft(green_channel, axis=0)
    frequencies = np.fft.fftfreq(green_channel.shape[0], d=1.0 / fps)

    print("fft shape", fft.shape)
    amplitude = np.abs(fft).mean(axis=(1, 2))
    # amplitude = np.abs(fft)[:, 0, 0]
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    # Plot the amplitude spectrum within the frequency range
    plt.figure()
    plt.plot(frequencies[mask], amplitude[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs Frequency")
    plt.grid(True)

    plt.savefig("freq.png")
    plt.close()


def getGaussianPyramids(images, kernel, level):
    # gaussian_pyramids = np.zeros_like(images, dtype=np.float32)
    gaussian_pyramids = np.zeros(
        (images.shape[0], images.shape[1], images.shape[2]), dtype=np.float32
    )

    for i in tqdm.tqdm(
        range(images.shape[0]), ascii=True, desc="Gaussian Pyramids Generation"
    ):
        green_channel = images[i, :, :, 1]

        gaussian_pyramids[i] = generateGaussianPyramid(
            image=green_channel, kernel=kernel, level=level
        )

    return gaussian_pyramids


def filterGaussianPyramids(pyramids, fps, freq_range, alpha, attenuation):

    filtered_pyramids = idealTemporalBandpassFilter(
        images=pyramids, fps=fps, freq_range=freq_range
    ).astype(np.float32)

    filtered_pyramids *= alpha
    # filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids
