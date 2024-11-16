from processing import loadVideo
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks, csd

from gaussian_pyramid import getGaussianPyramids, filterGaussianPyramids
from constants import gaussian_kernel

from multiprocessing import Pool

images, fps = loadVideo(video_path="../data/face.mp4")

pyramids = getGaussianPyramids(images, gaussian_kernel, 3)
freq_range = (0.83, 1)
blurred_images = filterGaussianPyramids(
    pyramids, fps, freq_range, alpha=2, attenuation=1
)  # TODO: check alpha
window_radius = 3
window_size = window_radius * 2 + 1


def select_center_point(image):
    selected_point = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point[0] = (y, x)
            print(f"Selected point: {selected_point[0]}")
            cv2.circle(param, (y, x), 5, (0, 255, 0), -1)
            cv2.imshow("Select Center Point", param)

    temp_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    cv2.imshow("Select Center Point", temp_image)
    cv2.setMouseCallback("Select Center Point", mouse_callback, temp_image)

    while selected_point[0] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if Esc is pressed
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return selected_point[0]


def draw_intensity(signal):

    plt.figure(figsize=(10, 5))
    plt.plot(signal, label="Green Channel Intensity", color="green")
    plt.xlabel("Frame")
    plt.ylabel("Average Intensity")
    plt.title("Normalized Green Channel Intensity Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("./out/intensity.png")
    plt.close()


def draw_fft(frequencies, fft, freq_range):

    amplitude = np.abs(fft)

    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    plt.figure()
    plt.plot(frequencies[mask], amplitude[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs. Frequency")
    plt.grid(True)
    plt.savefig("./out/freq.png")
    plt.close()


def draw_psd(signal, fs=30.0):
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]  # Convert to 2D with one row

    plt.figure(figsize=(10, 6))

    for idx, row in enumerate(signal):
        freqs, psd = welch(row, fs=fs, nperseg=256)

        plt.plot(freqs, psd, label=f"Signal {idx + 1}")

    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.grid()
    plt.legend()
    plt.savefig("./out/psd.png")


def get_snr(signal, fs, freq_range):
    N = len(signal)
    freq_domain = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    freq_domain = freq_domain[pos_mask]

    power_spectrum = np.abs(freq_domain) ** 2 / N

    signal_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    signal_power = np.sum(power_spectrum[signal_mask])

    noise_mask = ~signal_mask
    noise_power = np.sum(power_spectrum[noise_mask])

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def draw_box(center_point, window_radius, s, boxed_video_path="./out/face.mp4"):
    boxed_video = cv2.VideoWriter(
        boxed_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (images.shape[2], images.shape[1]),
    )

    for i in range(images.shape[0]):
        frame = images[i].astype(np.uint8)
        top_left = (center_point[1] - window_radius, center_point[0] - window_radius)
        bottom_right = (
            center_point[1] + window_radius,
            center_point[0] + window_radius,
        )
        boxed_frame = cv2.rectangle(
            frame.copy(), top_left, bottom_right, (0, 255, 0), 1
        )

        signal_height = 50  # height of the signal overlay area
        signal_width = frame.shape[1]
        signal_overlay = np.zeros((signal_height, signal_width, 3), dtype=np.uint8)

        normalized_signal = (
            (s - np.min(s)) / (np.max(s) - np.min(s)) * (signal_height - 1)
        )

        # draw the signal line
        for j in range(1, len(normalized_signal)):
            if j < i:
                cv2.line(
                    signal_overlay,
                    (j - 1, signal_height - int(normalized_signal[j - 1])),
                    (j, signal_height - int(normalized_signal[j])),
                    (0, 255, 0),
                    2,
                )

        # overlay positioning
        boxed_frame[-signal_height:, :] = cv2.addWeighted(
            boxed_frame[-signal_height:, :], 0.5, signal_overlay, 0.5, 0
        )

        window_size = int(3 * fps)
        start_idx = max(0, i - window_size)
        window_signal = s[start_idx:i]

        peaks, _ = find_peaks(window_signal, distance=fps / 2)
        if len(peaks) > 1:
            peak_times = peaks / fps
            rr_intervals = np.diff(peak_times)
            avg_rr_interval = np.mean(rr_intervals)
            bpm = 60 / avg_rr_interval
        else:
            bpm = 0

        cv2.putText(
            boxed_frame,
            f"BPM: {int(bpm)}",
            (200, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,  # font scale
            (0, 255, 0),
            3,  # thickness
            cv2.LINE_AA,
        )

        boxed_video.write(cv2.cvtColor(boxed_frame, cv2.COLOR_RGB2BGR))

    boxed_video.release()
    print(f"Boxed region video saved to {boxed_video_path}")


def get_signal_from_rgb(patch_images):
    r = patch_images[:, :, :, 0].astype(np.float32)
    g = patch_images[:, :, :, 1].astype(np.float32)
    b = patch_images[:, :, :, 2].astype(np.float32)

    norm = np.sqrt(r**2 + g**2 + b**2)

    r_n = (r / norm).mean(axis=(1, 2))
    g_n = (g / norm).mean(axis=(1, 2))
    b_n = (b / norm).mean(axis=(1, 2))

    x_s = 3 * r_n - 2 * g_n
    y_s = 1.5 * r_n + g_n - 1.5 * b_n

    alpha = np.std(x_s) / np.std(y_s)

    s = x_s - alpha * y_s

    # F, T, Z = stft(s, 30)
    # # Z = np.squeeze(Z, axis=0)

    # band = np.argwhere((F > freq_range[0]) & (F < freq_range[1])).flatten()
    # spect = np.abs(Z[band, :])  # spectrum magnitude
    # freqs = 60 * F[band]  # spectrum freq in bpm

    # bpm = freqs[np.argmax(spect, axis=0)]
    # print("bpm is ", bpm)

    # draw_psd(s)
    # draw_box(center_point, window_radius, s)

    return s


def create_heatmap_video(images, time_delays, fps, signal_ref, valid_mask):
    num_frames, height, width, _ = images.shape
    n_patches_h, n_patches_w = time_delays.shape

    patch_height = height // n_patches_h
    patch_width = width // n_patches_w

    heatmap_frames = []

    min_val = np.min(signal_ref)
    max_val = np.max(signal_ref)

    if max_val == min_val:
        signal_ref = np.zeros_like(signal_ref)
    else:
        signal_ref = (signal_ref - min_val) / (max_val - min_val)  # range from 0 to 1

    for t in range(num_frames):
        heatmap = np.zeros((height, width), dtype=np.float32)

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y_start = i * patch_height
                y_end = y_start + patch_height
                x_start = j * patch_width
                x_end = x_start + patch_width

                if valid_mask[i, j]:
                    time_delay = time_delays[i, j]

                    sample_index = int(time_delay * fps)
                    if sample_index < len(signal_ref):
                        amplitude = signal_ref[(t + sample_index) % len(signal_ref)]
                    else:
                        amplitude = 0

                    heatmap[y_start:y_end, x_start:x_end] = amplitude
                else:
                    heatmap[y_start:y_end, x_start:x_end] = 0
        heatmap_normalized = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        heatmap_frames.append(heatmap_color)

    video_filename = "heatmap_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame in heatmap_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Heatmap video saved as {video_filename}")


def process_patch(args):
    i, j = args
    patch_images = blurred_images[
        :,
        i * window_size : (i + 1) * window_size,
        j * window_size : (j + 1) * window_size,
        :,
    ]
    result = get_signal_from_rgb(patch_images)
    return (i, j, result)


def process_video(
    images,
    freq_range,
    fps=30,
):

    # center_point = (114, 256)

    # n_frames = images.shape[0]
    # images_hsv = np.empty_like(images)
    # for i in range(n_frames):
    #     images_hsv[i, :, :] = cv2.cvtColor(images[i, :, :], cv2.COLOR_BGR2HSV)

    center_point = select_center_point(images[0])

    height, width = images.shape[1], images.shape[2]
    n_frames = images.shape[0]

    n_patches_h = height // window_size
    n_patches_w = width // window_size

    center_i = center_point[0] // window_size
    center_j = center_point[1] // window_size

    s_list = np.zeros((n_patches_h, n_patches_w, n_frames))

    tasks = [(i, j) for i in range(n_patches_h) for j in range(n_patches_w)]

    with Pool() as pool:
        results = pool.map(process_patch, tasks)

    for i, j, result in results:
        s_list[i, j, :] = result

    # for i in range(n_patches_h):
    #     for j in range(n_patches_w):
    #         patch_images = images[
    #             :,
    #             i * window_size : (i + 1) * window_size,
    #             j * window_size : (j + 1) * window_size,
    #             :,
    #         ]
    #         s_list[i, j, :] = get_signal_from_rgb(patch_images, freq_range)

    s_center = s_list[center_i, center_j, :]
    snr_all = []
    snr_all = np.zeros((n_patches_h, n_patches_w))
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            snr_all[i, j] = get_snr(s_list[i, j, :], fps, freq_range)

    valid_mask = snr_all > (np.mean(snr_all) - np.std(snr_all))

    time_delays = np.zeros((n_patches_h, n_patches_w))
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            s_patch = s_list[i, j, :]
            f, Pxy = csd(s_patch, s_center, fs=fps, nperseg=256)
            idx = np.argmax(np.abs(Pxy))
            phase_spectrum = np.unwrap(np.angle(Pxy))
            phase_diff = phase_spectrum[idx]
            time_delay = phase_diff / (2 * np.pi * f[idx])
            time_delays[i, j] = time_delay

    min_delay = np.min(time_delays)
    max_delay = np.max(time_delays)

    if max_delay - min_delay == 0:
        normalized_delays = np.zeros_like(time_delays, dtype=np.uint8)
    else:
        normalized_delays = (time_delays - min_delay) / (max_delay - min_delay)
        normalized_delays = (normalized_delays * 255).astype(np.uint8)

    jet_colormap = cv2.applyColorMap(normalized_delays, cv2.COLORMAP_JET)

    # jet_colormap = cv2.resize(jet_colormap, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite("time_delays_colormap.png", jet_colormap)

    # print(time_delays)
    create_heatmap_video(images, time_delays, fps, s_center, valid_mask)

    # # Plot the heat map of time delays
    # plt.imshow(time_delays, cmap="hot", interpolation="nearest")
    # plt.colorbar(label="Time Delay (s)")
    # plt.title("Time Delay Heat Map")
    # plt.show()


process_video(images, freq_range=freq_range, fps=fps)
