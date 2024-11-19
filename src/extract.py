from matplotlib.lines import segment_hits
from processing import loadVideo
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
from scipy.signal import csd, welch
from multiprocessing import Pool, cpu_count

from gaussian_pyramid import getGaussianPyramids, filterGaussianPyramids
from constants import gaussian_kernel

from utils import select_center_point, select_segmenting_mask
from methods import get_chrom_signal, get_green_signal
from visual import draw_box, write_video
from skimage.restoration import unwrap_phase


class Pipeline:
    def __init__(self, video_path):
        self.video, self.fps = loadVideo(video_path=video_path)
        self.n_frames, self.height, self.width, _ = self.video.shape
        self.window_size = 1 * 2 + 1
        self.n_patches_h = self.height // self.window_size
        self.n_patches_w = self.width // self.window_size
        self.segmentation_mask = select_segmenting_mask(self.video[0])
        new_height = self.n_patches_h * self.window_size
        new_width = self.n_patches_w * self.window_size
        mask_cropped = self.segmentation_mask[:new_height, :new_width]

        mask_reshaped = mask_cropped.reshape(
            self.n_patches_h, self.window_size, self.n_patches_w, self.window_size
        )
        self.patch_segmentation_mask = mask_reshaped.all(axis=(1, 3))
        print(self.patch_segmentation_mask)
        print(self.segmentation_mask)
        print(self.patch_segmentation_mask.shape)
        print(self.segmentation_mask.shape)

        masked_image = self.video[0].copy()
        masked_image[~self.segmentation_mask] = [0, 0, 0]
        print(masked_image.shape)
        self.center_point = select_center_point(np.array(masked_image))

        print("Calculating heart rate")
        self.calc_heart_rate()
        print("Finsihed heart rate")

        print("Calculating signal map")
        self.calc_signals_map()
        print("Finished signal map")

        print("Calculating valid mask")
        self.calc_valid_mask()
        print("Finsihed valid mask")

        print("Calculating signal ref")
        self.calc_signal_ref()
        print("FInsih signal ref")

        print("Calculating timedelay")
        self.calc_time_delays()
        print("FInsih timedelay")

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

    def spatial_filter_video(self, freq_range):
        pyramids = getGaussianPyramids(self.video, gaussian_kernel, 3)
        blurred_video = filterGaussianPyramids(
            pyramids, self.fps, freq_range, alpha=2, attenuation=1
        )  # TODO: check alpha
        return blurred_video

    def calc_valid_mask(self):
        heart_rate_freq = self.heart_rate / 60  # in Hz
        heart_rate_range = (heart_rate_freq - 0.15, heart_rate_freq + 0.15)

        snr_all = np.zeros((self.n_patches_h, self.n_patches_w))

        valid_indices = np.where(self.patch_segmentation_mask)
        for i, j in zip(valid_indices[0], valid_indices[1]):
            snr_all[i, j] = self.get_snr(
                self.s_list[i, j, :], self.fps, heart_rate_range
            )

        threshold = np.nanmean(snr_all) - 3 * np.nanstd(snr_all)

        self.valid_mask = (snr_all > threshold) & (~np.isnan(snr_all))

    def calc_signal_ref(self, neighborhood_size=1):
        center_i = self.center_point[0] // self.window_size
        center_j = self.center_point[1] // self.window_size

        i_start = max(center_i - neighborhood_size, 0)
        i_end = min(center_i + neighborhood_size + 1, self.n_patches_h)
        j_start = max(center_j - neighborhood_size, 0)
        j_end = min(center_j + neighborhood_size + 1, self.n_patches_w)

        valid_signals = []

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if self.patch_segmentation_mask[i, j]:
                    valid_signals.append(self.s_list[i, j, :])

        if not valid_signals:
            raise ValueError("No valid neighboring patches found for signal reference.")

        signal_ref = np.mean(valid_signals, axis=0)

        min_val = np.min(signal_ref)
        max_val = np.max(signal_ref)

        if max_val == min_val:
            signal_ref = np.zeros_like(signal_ref)
        else:
            signal_ref = (signal_ref - min_val) / (max_val - min_val)

        self.signal_ref = signal_ref

    def calc_heart_rate(self):
        signal = get_chrom_signal(  # NOTE: Chrom might work better
            self.video[
                :,
                self.center_point[0] - 10 : self.center_point[0] + 10,
                self.center_point[1] - 10 : self.center_point[1] + 10,
                :,
            ]
        )

        freq_range = (0.5, 3.333)
        freqs, psd = welch(signal, fs=self.fps, nperseg=256)

        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

        freqs_in_range = freqs[mask]
        psd_in_range = psd[mask]

        if len(freqs_in_range) == 0:
            raise ValueError("No frequencies found in the specified range.")

        idx = np.argmax(psd_in_range)

        prominent_freq = freqs_in_range[idx]

        plt.figure(figsize=(10, 6))
        plt.plot(freqs, psd, label=f"Signal {0 + 1}")
        plt.title("Power Spectral Density (PSD)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.grid()
        plt.legend()
        plt.savefig("./out/psd.png")

        self.heart_rate = prominent_freq * 60
        print("guessed bpm=", self.heart_rate)

    def show_heart_rates(self):
        draw_box(
            self.video, self.fps, self.center_point, self.window_size, self.signal_ref
        )

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

        if noise_power == 0 or signal_power == 0:
            print("Noise or signal power is 0")
            snr = np.nan
        else:
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
            (i, j)
            for i in range(self.n_patches_h)
            for j in range(self.n_patches_w)
            if self.patch_segmentation_mask[i, j]
        ]

        with Pool(
            initializer=self.init_pool_processes,
            initargs=(filtered_video, self.window_size),
            processes=cpu_count(),
        ) as pool:
            results = pool.map(self.process_patch, tasks)
        for i, j, result in results:
            s_list[i, j, :] = result
        self.s_list = s_list

    @staticmethod
    def init_pool_processes_time_delay(s_list_, signal_ref_, fps_):
        global s_list
        global signal_ref
        global fps
        s_list = s_list_
        signal_ref = signal_ref_
        fps = fps_

    @staticmethod
    def _compute_time_delay(args):
        i, j = args
        s_patch = s_list[i, j, :]
        f, Pxy = csd(s_patch, signal_ref, fs=fps, nperseg=256)
        idx = np.argmax(np.abs(Pxy))
        phase_spectrum = np.unwrap(np.angle(Pxy))
        phase_diff = phase_spectrum[idx]
        time_delay = phase_diff / (2 * np.pi * f[idx])
        return (i, j, time_delay)

    def calc_time_delays(self, chunksize=100):

        time_delays = np.zeros((self.n_patches_h, self.n_patches_w))

        indices = [
            (i, j)
            for i in range(self.n_patches_h)
            for j in range(self.n_patches_w)
            if self.valid_mask[i, j] and self.patch_segmentation_mask[i, j]
        ]

        with Pool(
            processes=cpu_count(),
            initializer=self.init_pool_processes_time_delay,
            initargs=(self.s_list, self.signal_ref, self.fps),
        ) as pool:
            results = pool.map(self._compute_time_delay, indices, chunksize)

        for i, j, delay in results:  # TODO: verify how time_delay can be out of range
            # if delay <= 0.3:
            #     time_delays[i, j] = delay
            # else:
            #     self.valid_mask[i, j] = False
            time_delays[i, j] = delay

        self.time_delays = time_delays

    # @staticmethod
    # def init_pool_processes_time_delay(s_list_, fps_, heart_rate_freq_):
    #     global s_list
    #     global fps
    #     global heart_rate_freq
    #     s_list = s_list_
    #     fps = fps_
    #     heart_rate_freq = heart_rate_freq_

    # @staticmethod
    # def _compute_phase_angle(args):
    #     i, j = args
    #     s_patch = s_list[i, j, :]
    #     N = len(s_patch)
    #     freq_domain = np.fft.fft(s_patch)
    #     freqs = np.fft.fftfreq(N, d=1 / fps)
    #     idx = np.argmin(np.abs(freqs - heart_rate_freq))
    #     phase_angle = np.angle(freq_domain[idx])
    #     return (i, j, phase_angle)

    # def calc_time_delays(self):
    #     heart_rate_freq = self.heart_rate / 60  # in Hz
    #     phase_angles = np.zeros((self.n_patches_h, self.n_patches_w))
    #     indices = [
    #         (i, j)
    #         for i in range(self.n_patches_h)
    #         for j in range(self.n_patches_w)
    #         if self.valid_mask[i, j]
    #     ]

    #     with Pool(
    #         processes=cpu_count(),
    #         initializer=self.init_pool_processes_time_delay,
    #         initargs=(self.s_list, self.fps, heart_rate_freq),
    #     ) as pool:
    #         results = pool.map(self._compute_phase_angle, indices)

    #     for i, j, phase_angle in results:
    #         phase_angles[i, j] = phase_angle

    #     ref_i, ref_j = np.unravel_index(
    #         np.nanargmax(phase_angles * self.valid_mask), phase_angles.shape
    #     )
    #     phi_ref = phase_angles[ref_i, ref_j]
    #     delta_phi = phi_ref - phase_angles

    #     delta_phi_unwrapped = unwrap_phase(delta_phi)

    #     delta_t = delta_phi_unwrapped / (2 * np.pi * heart_rate_freq)
    #     # self.time_delays = np.where(np.abs(delta_t) <= 0.3, delta_t, np.nan)
    #     self.time_delays = delta_t
    #
    # @staticmethod
    # def init_pool_processes_time_delay(s_list_, signal_ref_, fps_, max_lag_frames_):
    #     global s_list
    #     global signal_ref
    #     global fps
    #     global max_lag_frames
    #     s_list = s_list_
    #     signal_ref = signal_ref_
    #     fps = fps_
    #     max_lag_frames = max_lag_frames_

    # @staticmethod
    # def _compute_time_delay(args):
    #     i, j = args
    #     s_patch = s_list[i, j, :]

    #     s_patch_centered = s_patch - np.mean(s_patch)
    #     signal_ref_centered = signal_ref - np.mean(signal_ref)
    #     correlation = np.correlate(s_patch_centered, signal_ref_centered, mode="full")
    #     N = len(s_patch)
    #     lags = np.arange(-N + 1, N)
    #     lag_mask = np.abs(lags) <= max_lag_frames
    #     correlation = correlation[lag_mask]
    #     lags = lags[lag_mask]
    #     if correlation.size == 0:
    #         delta_t = np.nan
    #     else:
    #         max_corr_index = np.argmax(correlation)
    #         max_lag = lags[max_corr_index]
    #         delta_t = max_lag / fps

    #     return (i, j, delta_t)

    # def calc_time_delays(self):
    #     time_delays = np.zeros((self.n_patches_h, self.n_patches_w))
    #     indices = [
    #         (i, j)
    #         for i in range(self.n_patches_h)
    #         for j in range(self.n_patches_w)
    #         if self.valid_mask[i, j]
    #     ]

    #     max_lag_seconds = 0.3
    #     max_lag_frames = int(max_lag_seconds * self.fps)

    #     with Pool(
    #         processes=cpu_count(),
    #         initializer=self.init_pool_processes_time_delay,
    #         initargs=(self.s_list, self.signal_ref, self.fps, max_lag_frames),
    #     ) as pool:
    #         results = pool.map(self._compute_time_delay, indices)

    #     for i, j, delta_t in results:
    #         time_delays[i, j] = delta_t

    #     self.time_delays = time_delays

    def get_heatmap_video(self):
        patch_height = self.height // self.n_patches_h
        patch_width = self.width // self.n_patches_w

        i_indices, j_indices = np.meshgrid(
            np.arange(self.n_patches_h), np.arange(self.n_patches_w), indexing="ij"
        )
        y_starts = i_indices * patch_height
        y_ends = (i_indices + 1) * patch_height
        x_starts = j_indices * patch_width
        x_ends = (j_indices + 1) * patch_width

        y_start_flat = y_starts.flatten()
        y_end_flat = y_ends.flatten()
        x_start_flat = x_starts.flatten()
        x_end_flat = x_ends.flatten()

        valid_segments = (self.valid_mask & self.patch_segmentation_mask).flatten()

        sample_indices = (self.time_delays.flatten() * self.fps).astype(int)
        sample_indices = sample_indices % len(self.signal_ref)

        frame_indices = np.arange(self.n_frames).reshape(-1, 1)  # (n_frames, 1)
        sample_indices_matrix = (frame_indices + sample_indices) % len(self.signal_ref)
        amplitudes = self.signal_ref[sample_indices_matrix]

        amplitudes[:, ~valid_segments] = 0

        heatmaps = np.zeros((self.n_frames, self.height, self.width), dtype=np.float32)

        for idx in range(len(y_start_flat)):
            y_start = y_start_flat[idx]
            y_end = y_end_flat[idx]
            x_start = x_start_flat[idx]
            x_end = x_end_flat[idx]
            heatmaps[:, y_start:y_end, x_start:x_end] = amplitudes[
                :, idx, np.newaxis, np.newaxis
            ]

        heatmaps_normalized = np.clip(heatmaps, 0.0, 1.0) * 255.0
        heatmaps_normalized = heatmaps_normalized.astype(np.uint8)

        heatmap_frames = np.empty(
            (self.n_frames, self.height, self.width, 3), dtype=np.uint8
        )
        for t in range(self.n_frames):
            heatmap_color = cv2.applyColorMap(heatmaps_normalized[t], cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            heatmap_frames[t] = heatmap_color_rgb

        return heatmap_frames

    def process_video(self):

        min_delay = np.nanmin(self.time_delays)
        max_delay = np.nanmax(self.time_delays)

        if max_delay - min_delay == 0:
            normalized_delays = np.zeros_like(self.time_delays, dtype=np.uint8)
        else:
            normalized_delays = (self.time_delays - min_delay) / (max_delay - min_delay)
            normalized_delays = (normalized_delays * 255).astype(np.uint8)

        jet_colormap = cv2.applyColorMap(normalized_delays, cv2.COLORMAP_JET)
        jet_colormap = cv2.resize(
            jet_colormap, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        cv2.imwrite("PTT.png", jet_colormap)

        print("getting heatmap frames")
        heatmap_frames = self.get_heatmap_video()
        print("finish getting heatmap frames")
        heatmap_float = heatmap_frames.astype(np.float32)
        red_channel = heatmap_float[..., 0]  # (n_frames, height, width)
        red_normalized = red_channel / 255.0
        mask = red_normalized > 0.05
        combined_mask = self.segmentation_mask & mask  # (n_frames, height, width)
        mask = combined_mask[..., np.newaxis].repeat(3, axis=-1)  # add channel dim
        overlaid_video = np.where(mask, heatmap_frames, self.video)
        write_video(overlaid_video, self.fps, "heatmap.avi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video for heart rate analysis."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    pipe = Pipeline(args.video_path)
    pipe.process_video()
    # pipe.show_heart_rates()
