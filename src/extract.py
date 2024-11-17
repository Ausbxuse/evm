from processing import loadVideo
import cv2
import numpy as np
from scipy.signal import csd

from gaussian_pyramid import getGaussianPyramids, filterGaussianPyramids
from constants import gaussian_kernel

from multiprocessing import Pool
from utils import select_center_point
from methods import get_chrom_signal
from visual import draw_box, write_video


class Pipeline:
    @staticmethod
    def init_pool_processes(filtered_video_, window_size_):
        global filtered_video
        global window_size
        filtered_video = filtered_video_
        window_size = window_size_

    @staticmethod
    def process_patch(args):
        i, j = args
        patch_images = filtered_video[
            :,
            i * window_size : (i + 1) * window_size,
            j * window_size : (j + 1) * window_size,
            :,
        ]
        result = get_chrom_signal(patch_images)
        return (i, j, result)

    def __init__(self, video_path):
        self.video, self.fps = loadVideo(video_path=video_path)
        self.n_frames, self.height, self.width, _ = self.video.shape
        self.window_size = 3 * 2 + 1
        self.n_patches_h = self.height // self.window_size
        self.n_patches_w = self.width // self.window_size
        self.center_point = np.array(select_center_point(self.video[0]))

        self.calc_heart_rate()
        print("Calculating signal map")
        self.calc_signals_map()
        print("Calculating time delays")
        self.calc_time_delays()

    def spatial_filter_video(self, freq_range):
        pyramids = getGaussianPyramids(self.video, gaussian_kernel, 3)
        blurred_video = filterGaussianPyramids(
            pyramids, self.fps, freq_range, alpha=2, attenuation=1
        )  # TODO: check alpha
        return blurred_video

    def calc_heart_rate(self):  # TODO: to be implemented
        # freq_range = (0.5, 3.333)  # from 30 bpm to 200 bpm
        # blurred_video = self.spatial_filter_video(freq_range)
        self.heart_rate = 55

    def get_snr(self, signal, fs, freq_range):
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

    def calc_signals_map(self):
        """
        returns a map of all signals specified by the window_size in the entire video.
        """
        heart_rate_freq = self.heart_rate / 60  # in Hz
        heart_rate_range = (heart_rate_freq - 0.15, heart_rate_freq + 0.15)
        filtered_video = self.spatial_filter_video(heart_rate_range)
        s_list = np.zeros((self.n_patches_h, self.n_patches_w, self.n_frames))
        tasks = [
            (i, j) for i in range(self.n_patches_h) for j in range(self.n_patches_w)
        ]

        with Pool(
            initializer=self.init_pool_processes,
            initargs=(filtered_video, self.window_size),
        ) as pool:
            results = pool.map(self.process_patch, tasks)
        for i, j, result in results:
            s_list[i, j, :] = result
        self.s_list = s_list

    def calc_time_delays(self):
        """
        user must select the reference point to calculate the time delay from for all the signals
        """

        center_i = self.center_point[0] // self.window_size
        center_j = self.center_point[1] // self.window_size

        s_center = self.s_list[center_i, center_j, :]

        time_delays = np.zeros((self.n_patches_h, self.n_patches_w))
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                s_patch = self.s_list[i, j, :]
                f, Pxy = csd(s_patch, s_center, fs=self.fps, nperseg=256)
                idx = np.argmax(np.abs(Pxy))
                phase_spectrum = np.unwrap(np.angle(Pxy))
                phase_diff = phase_spectrum[idx]
                time_delay = phase_diff / (2 * np.pi * f[idx])
                time_delays[i, j] = time_delay
        self.time_delays = time_delays

    def get_heatmap_video(self, valid_mask):
        center_i = self.center_point[0] // self.window_size
        center_j = self.center_point[1] // self.window_size
        signal_ref = self.s_list[center_i, center_j, :]
        min_val = np.min(signal_ref)
        max_val = np.max(signal_ref)

        patch_height = self.height // self.n_patches_h
        patch_width = self.width // self.n_patches_w

        heatmap_frames = []

        if max_val == min_val:
            signal_ref = np.zeros_like(signal_ref)
        else:
            signal_ref = (signal_ref - min_val) / (
                max_val - min_val
            )  # range from 0 to 1

        for t in range(self.n_frames):
            heatmap = np.zeros((self.height, self.width), dtype=np.float32)

            for i in range(self.n_patches_h):
                for j in range(self.n_patches_w):
                    y_start = i * patch_height
                    y_end = y_start + patch_height
                    x_start = j * patch_width
                    x_end = x_start + patch_width

                    if valid_mask[i, j]:
                        time_delay = self.time_delays[i, j]
                        sample_index = int(time_delay * self.fps)
                        if sample_index < len(signal_ref):
                            amplitude = signal_ref[(t + sample_index) % len(signal_ref)]
                        else:
                            amplitude = 255
                        heatmap[y_start:y_end, x_start:x_end] = amplitude
                    else:
                        heatmap[y_start:y_end, x_start:x_end] = 0
            heatmap_normalized = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            heatmap_frames.append(heatmap_color)

        return heatmap_frames

    def process_video(self):
        heart_rate_freq = self.heart_rate / 60  # in Hz
        heart_rate_range = (heart_rate_freq - 0.15, heart_rate_freq + 0.15)

        snr_all = []
        snr_all = np.zeros((self.n_patches_h, self.n_patches_w))
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                snr_all[i, j] = self.get_snr(
                    self.s_list[i, j, :], self.fps, heart_rate_range
                )

        valid_mask = snr_all > (np.mean(snr_all) - np.std(snr_all))

        min_delay = np.min(self.time_delays)
        max_delay = np.max(self.time_delays)

        if max_delay - min_delay == 0:
            normalized_delays = np.zeros_like(self.time_delays, dtype=np.uint8)
        else:
            normalized_delays = (self.time_delays - min_delay) / (max_delay - min_delay)
            normalized_delays = (normalized_delays * 255).astype(np.uint8)

        jet_colormap = cv2.applyColorMap(normalized_delays, cv2.COLORMAP_JET)
        jet_colormap = cv2.resize(
            jet_colormap, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        cv2.imwrite("time_delays_colormap.png", jet_colormap)

        write_video(
            self.get_heatmap_video(valid_mask),
            self.fps,
            self.width,
            self.height,
            "heatmap.avi",
        )

        # # Plot the heat map of time delays
        # plt.imshow(time_delays, cmap="hot", interpolation="nearest")
        # plt.colorbar(label="Time Delay (s)")
        # plt.title("Time Delay Heat Map")
        # plt.show()


if __name__ == "__main__":
    pipe = Pipeline("../data/face.mp4")
    pipe.process_video()
