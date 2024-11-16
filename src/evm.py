import argparse
import os

from constants import gaussian_kernel
from gaussian_pyramid import (
    filterGaussianPyramids,
    getGaussianPyramids,
    get_frequency_map,
)
from processing import (
    getGaussianOutputVideo,
    loadVideo,
    saveVideo,
)


def gaussian_evm(images, fps, kernel, level, alpha, freq_range, attenuation):

    get_frequency_map(images, freq_range=freq_range, fps=fps)

    gaussian_pyramids = getGaussianPyramids(images=images, kernel=kernel, level=level)

    print("Gaussian Pyramids Filtering...")
    filtered_pyramids = filterGaussianPyramids(
        pyramids=gaussian_pyramids,
        fps=fps,
        freq_range=freq_range,
        alpha=alpha,
        attenuation=attenuation,
    )
    print("Finished!")

    output_video = getGaussianOutputVideo(
        original_images=images, filtered_images=filtered_pyramids
    )

    return output_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification for colors and motions magnification"
    )

    parser.add_argument(
        "--video_path",
        "-v",
        type=str,
        help="Path to the video to be used",
        required=True,
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="Number of level of the Gaussian Pyramid",
        required=False,
        default=4,
    )

    parser.add_argument(
        "--alpha",
        "-a",
        type=int,
        help="Amplification factor",
        required=False,
        default=100,
    )

    parser.add_argument(
        "--low_omega",
        "-lo",
        type=float,
        help="Minimum allowed frequency",
        required=False,
        default=0.833,
    )

    parser.add_argument(
        "--high_omega",
        "-ho",
        type=float,
        help="Maximum allowed frequency",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--saving_path",
        "-s",
        type=str,
        help="Saving path of the magnified video",
        required=True,
    )

    parser.add_argument(
        "--attenuation",
        "-at",
        type=float,
        help="Attenuation factor for I and Q channel post filtering",
        required=False,
        default=1,
    )

    args = parser.parse_args()
    kwargs = {}
    kwargs["kernel"] = gaussian_kernel
    kwargs["level"] = args.level
    kwargs["alpha"] = args.alpha
    kwargs["freq_range"] = [args.low_omega, args.high_omega]
    kwargs["attenuation"] = args.attenuation
    video_path = args.video_path

    assert os.path.exists(video_path), f"Video {video_path} not found :("

    images, fps = loadVideo(video_path=video_path)
    kwargs["images"] = images
    kwargs["fps"] = fps

    output_video = gaussian_evm(**kwargs)

    saveVideo(video=output_video, saving_path=args.saving_path, fps=fps)
