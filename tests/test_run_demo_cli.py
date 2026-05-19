from __future__ import annotations

from ind_vias_perception.apps.run_demo import build_parser


def test_cli_parser_accepts_image_video_output_show_and_max_frames():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--image",
            "frame.jpg",
            "--video",
            "clip.mp4",
            "--output",
            "annotated.mp4",
            "--show",
            "--debug-overlay",
            "--max-frames",
            "7",
        ]
    )

    assert args.image == "frame.jpg"
    assert args.video == "clip.mp4"
    assert args.output == "annotated.mp4"
    assert args.show is True
    assert args.debug_overlay is True
    assert args.max_frames == 7
