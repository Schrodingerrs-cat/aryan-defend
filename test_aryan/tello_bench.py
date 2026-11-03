"""Benchmark the Tello link: video fps/stalls, command round-trip, rc send rate, state rate,
and (with --qr) end-to-end video latency. Never takes off.

  python3 tello_bench.py              # 15 s link benchmark, drone can sit anywhere
  python3 tello_bench.py --qr         # + video latency: point the drone camera at THIS laptop's
                                      #   screen showing the QR window (fill the camera view with it)
  python3 tello_bench.py --seconds 30
"""
import argparse
import socket
import threading
import time

import cv2
import numpy as np

from tello_stream import TELLO_ADDR, TelloStream


def stats(name, xs, unit):
    if len(xs) == 0:
        print(f"{name:<28} no data")
        return
    a = np.asarray(xs, dtype=float)
    print(f"{name:<28} n={len(a):<5} mean {a.mean():7.1f}  median {np.median(a):7.1f}  "
          f"p95 {np.percentile(a, 95):7.1f}  max {a.max():7.1f} {unit}")


def qr_image(text, size=560):
    enc = cv2.QRCodeEncoder.create()
    m = enc.encode(text)
    m = cv2.copyMakeBorder(m, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def state_listener(times, stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 8890))
    s.settimeout(0.5)
    while not stop.is_set():
        try:
            s.recvfrom(1024)
            times.append(time.perf_counter())
        except socket.timeout:
            pass
    s.close()


def rtt_loop(rtts, lost, stop):
    """Round-trip of 'battery?' on its own socket, ~10 per second."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 9001))
    s.settimeout(1.0)
    while not stop.is_set():
        t0 = time.perf_counter()
        s.sendto(b"battery?", TELLO_ADDR)
        try:
            s.recvfrom(256)
            rtts.append((time.perf_counter() - t0) * 1000)
        except socket.timeout:
            lost.append(1)
        time.sleep(0.1)
    s.close()


def rc_loop(intervals, stop, hz=20):
    """Send neutral rc at a target rate and record the real send intervals (jitter)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / hz
    nxt = time.perf_counter()
    last = None
    while not stop.is_set():
        s.sendto(b"rc 0 0 0 0", TELLO_ADDR)
        now = time.perf_counter()
        if last is not None:
            intervals.append((now - last) * 1000)
        last = now
        nxt += period
        time.sleep(max(0, nxt - time.perf_counter()))
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--qr", action="store_true", help="measure video latency via on-screen QR clock")
    args = ap.parse_args()

    tello = TelloStream()
    tello.start()  # command + streamon + video thread

    stop = threading.Event()
    state_t, rtts, lost, rc_iv = [], [], [], []
    for fn, a in ((state_listener, (state_t, stop)), (rtt_loop, (rtts, lost, stop)),
                  (rc_loop, (rc_iv, stop))):
        threading.Thread(target=fn, args=a, daemon=True).start()

    detector = cv2.QRCodeDetector() if args.qr else None
    frame_t, latencies = [], []
    last_id = 0
    print(f"Benchmarking for {args.seconds:.0f} s ..." + ("  (show the camera the QR window)" if args.qr else ""))
    t_end = time.time() + args.seconds + 3  # first 3 s is warm-up for the video pipeline
    t_start = time.time() + 3
    try:
        while time.time() < t_end:
            if args.qr:
                cv2.imshow("QR clock (point the Tello camera here)", qr_image(f"{time.time():.3f}"))
            item = tello.read_new(last_id)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if item is None:
                continue
            frame, last_id, t_recv = item
            if time.time() < t_start:
                continue
            frame_t.append(t_recv)
            if args.qr:
                text, _, _ = detector.detectAndDecode(frame)
                try:
                    latencies.append((t_recv - float(text)) * 1000)
                except ValueError:
                    pass
    finally:
        stop.set()
        tello.stop()
        cv2.destroyAllWindows()

    ft = np.asarray(frame_t)
    gaps = np.diff(ft) * 1000 if len(ft) > 1 else []
    dur = (ft[-1] - ft[0]) if len(ft) > 1 else 0
    st = np.diff(state_t) * 1000 if len(state_t) > 1 else []

    print("\n=== Tello link report ===")
    print(f"video fps                    {len(ft) / dur:.1f}" if dur else "video fps                    no frames!")
    stats("video frame gap", gaps, "ms")
    if len(gaps):
        print(f"video stalls (>150 ms)       {int((np.asarray(gaps) > 150).sum())}")
    stats("command RTT (battery?)", rtts, "ms")
    print(f"command packets lost         {len(lost)} of {len(rtts) + len(lost)}")
    stats("rc send interval (20 Hz)", rc_iv, "ms")
    if len(rc_iv):
        print(f"rc effective rate            {1000 / np.mean(rc_iv):.1f} Hz (target 20)")
    stats("state packet gap (~100 ms)", st, "ms")
    if len(st):
        print(f"state rate                   {1000 / np.mean(st):.1f} Hz")
    if args.qr:
        stats("video latency (QR)", latencies, "ms")
        if not latencies:
            print("  no QR decoded: fill the camera view with the QR window, steady, screen brightness up")
        else:
            print("  latency = frame arrival at python minus time shown on screen; includes screen "
                  "refresh (~16 ms) + camera exposure + decode")


if __name__ == "__main__":
    main()
