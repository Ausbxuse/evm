import numpy as np


def get_chrom_signal(rgb_video):
    """
    Takes an rgb image of shape (n_frames, height, width, n_channels) and output chrominance signal of shape (n_frames,)
    """
    r = rgb_video[:, :, :, 0].astype(np.float32)
    g = rgb_video[:, :, :, 1].astype(np.float32)
    b = rgb_video[:, :, :, 2].astype(np.float32)

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


def get_green_signal(rgb_video):
    r = rgb_video[:, :, :, 0].astype(np.float32)
    g = rgb_video[:, :, :, 1].astype(np.float32)
    b = rgb_video[:, :, :, 2].astype(np.float32)
    norm = np.sqrt(r**2 + g**2 + b**2)
    g_n = (g / norm).mean(axis=(1, 2))
    s = g_n

    return s
