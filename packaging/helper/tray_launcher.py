"""Packaged Inference Helper entry point: background server + optional system tray."""

from __future__ import annotations

import multiprocessing
import sys


def _run_server() -> None:
    from inference.helper.__main__ import main

    main()


def _tray_icon_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), fill=(16, 120, 88, 255))
    draw.rectangle((24, 28, 40, 36), fill=(255, 255, 255, 255))
    return image


def _run_with_tray() -> None:
    import pystray

    process = multiprocessing.Process(target=_run_server, daemon=True)
    process.start()

    def on_quit(icon, _item) -> None:  # noqa: ANN001
        icon.stop()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    icon = pystray.Icon(
        "nomicous-inference-helper",
        _tray_icon_image(),
        "Nomicous Inference Helper",
        menu=pystray.Menu(pystray.MenuItem("Quit helper", on_quit)),
    )
    try:
        icon.run()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def main() -> None:
    if "--no-tray" in sys.argv:
        _run_server()
        return

    try:
        import pystray  # noqa: F401
    except ImportError:
        _run_server()
        return

    # Give the child process a moment to bind before showing the tray icon.
    _run_with_tray()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
