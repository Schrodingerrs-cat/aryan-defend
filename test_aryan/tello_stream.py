"""Minimal Python 3 Tello helper: command socket + live video via OpenCV/FFMPEG.

Replaces the SDK's Python 2 / libh264decoder pipeline. Never takes off.
"""
import os
import socket
import threading
import time

# Must be set before cv2 opens the stream: drop late packets instead of stalling.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")

import cv2

from dongle import check, dev_socket, resolve

TELLO_ADDR = ("192.168.10.1", 8889)
VIDEO_URL = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"


class TelloStream:
    """One Tello. With `drone="TELLO-1"` every socket is pinned to that drone's USB dongle (see dongle.py), so several
    drones (all 192.168.10.1) can run on one PC; the video is forwarded to a per-drone localhost port for OpenCV."""

    def __init__(self, drone=None, skip_check=False):
        self.drone, self.iface, self.entry = drone, None, None
        self.video_url = VIDEO_URL
        if drone:
            self.entry = resolve(drone)
            self.iface = self.entry["iface"]
            problems = [] if skip_check else check(self.entry)
            if problems:
                raise RuntimeError(f"{self.entry['drone']}: " + "; ".join(problems))
            self.sock = dev_socket(self.iface, 9000, timeout=5)           # local port 9000 for replies
            self._video_sock = dev_socket(self.iface, 11111, timeout=1)  # the drone streams to <our ip>:11111
            self._fwd_port = int(self.entry["video_port"])
            self.video_url = f"udp://127.0.0.1:{self._fwd_port}?overrun_nonfatal=1&fifo_size=5000000"
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", 9000))  # local port 9000 for replies
            self.sock.settimeout(5)
        self.frame = None
        self.frame_id = 0      # increments per decoded frame
        self.frame_time = 0.0  # wall-clock time the latest frame arrived
        self._lock = threading.Lock()
        self._running = False

    def send(self, cmd):
        self.sock.sendto(cmd.encode("ascii"), TELLO_ADDR)
        try:
            data, addr = self.sock.recvfrom(2048)
        except socket.timeout:
            return None
        return data.decode("ascii", errors="ignore").strip()

    def send_nowait(self, cmd):
        """Fire-and-forget (rc commands get no reply)."""
        self.sock.sendto(cmd.encode("ascii"), TELLO_ADDR)

    def start(self):
        if self.send("command") != "ok":
            raise RuntimeError(
                "No reply from Tello. Are you on the TELLO-XXXXXX wifi and is the drone on?"
            )
        print("Battery:", self.send("battery?"), "%")
        if self.send("streamon") != "ok":
            raise RuntimeError("streamon failed")
        self._running = True
        if self.iface:
            threading.Thread(target=self._forward_video, daemon=True).start()
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _forward_video(self):
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while self._running:
            try:
                data, _ = self._video_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            out.sendto(data, ("127.0.0.1", self._fwd_port))

    def _read_loop(self):
        cap = cv2.VideoCapture(self.video_url, cv2.CAP_FFMPEG)
        while self._running:
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.frame_time = time.time()
            else:
                time.sleep(0.01)
        cap.release()

    def _keepalive(self):
        # Tello drops back to idle if it hears nothing for ~15 s.
        while self._running:
            time.sleep(5)
            self.sock.sendto(b"command", TELLO_ADDR)
            try:
                self.sock.recvfrom(2048)
            except socket.timeout:
                pass

    def read(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def read_new(self, last_id):
        """Return (frame, id, time) if a frame newer than last_id exists, else None."""
        with self._lock:
            if self.frame is None or self.frame_id == last_id:
                return None
            return self.frame.copy(), self.frame_id, self.frame_time

    def stop(self):
        self._running = False
        try:
            self.send("streamoff")
        finally:
            self.sock.close()
            if self.iface:
                self._video_sock.close()
