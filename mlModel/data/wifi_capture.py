#!/usr/bin/env python3
"""
WiFi image capture client for ESP32
Captures RGB565 96x96 frames and saves as PNG

Requirements:
    pip install requests pillow numpy

Usage:
    python3 wifi_capture.py
    Press ENTER to capture, Ctrl+C to stop.
"""

import os
import requests
import numpy as np
from PIL import Image
from datetime import datetime

ESP32_URL = "http://192.168.129.146/capture"
OUTPUT_DIR ="captures/"
IMG_SIZE = 96

frame_idx = 0


def rgb565_to_png(raw_bytes, filename):
    """Convert raw RGB565 bytes to a PNG file."""
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    pixels = data.reshape(-1, 2)

    rgb565 = (pixels[:, 0].astype(np.uint16) << 8) | pixels[:, 1].astype(np.uint16)

    r = ((rgb565 >> 11) & 0x1F).astype(np.uint8)
    g = ((rgb565 >> 5) & 0x3F).astype(np.uint8)
    b = (rgb565 & 0x1F).astype(np.uint8)

    # Expand to 8 bits
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)

    rgb = np.stack([r, g, b], axis=1).reshape(IMG_SIZE, IMG_SIZE, 3)
    Image.fromarray(rgb.astype(np.uint8)).save(filename)


def capture():
    global frame_idx
    print("[→] Requesting capture...")
    try:
        r = requests.get(ESP32_URL, timeout=10)
        if r.status_code == 200:
            expected = IMG_SIZE * IMG_SIZE * 2
            if len(r.content) != expected:
                print(f"[ERR] Expected {expected} bytes, got {len(r.content)}")
                return
            frame_idx += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"{timestamp}_{frame_idx:03d}.png")
            rgb565_to_png(r.content, filename)
            print(f"[✓] Saved → {filename}")
        else:
            print(f"[ERR] ESP32 returned {r.status_code}: {r.text}")
    except requests.exceptions.Timeout:
        print("[ERR] Request timed out")
    except requests.exceptions.ConnectionError:
        print("[ERR] Could not connect to ESP32")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"ESP32 at {ESP32_URL}")
    print("Press ENTER to capture, Ctrl+C to stop.\n")

    try:
        while True:
            input()
            capture()
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
