"""Interactive 3D viewer for a localized recording: the video plays next to a rotatable 3D scene.

  python3 viewer_3d.py data/20261023_201826            # build + open in the browser
  python3 viewer_3d.py data/<a> data/<b> --no-open
  python3 viewer_3d.py data/<session> --map maps/my_map.yaml --floor-y -1.22

Needs data/<session>/localization/localization.csv (run localization_pipeline.py first).
Writes data/<session>/localization/viewer_3d.html + viewer_video.mp4 (H.264 copy; browsers can't play the
MPEG-4 files OpenCV writes) and opens the page. Everything is local/offline.

Scene (metres; Z = height above the floor, the tag-10 wall at Y = 0, the room toward +Y, +X = right of tag 10):
  oriented squares for every mapped tag (red = origin tag 10, blue = wall tag, orange = floor tag), the floor and
  wall, the whole path (coloured by time), and for the current video frame: the drone (green = tag fix,
  red = lost / holding), its trail, the camera's view cone, lines to the tags it is using and a drop line to the floor.
Play the video (or drag its timeline) and the drone moves; click the path to jump the video to that moment.
Drag the 3D scene to rotate, scroll to zoom. Open with  ...viewer_3d.html#t=60  to start at 60 s.

View-cone direction: heading from the tag pose, pitch from the Tello's IMU (nose-up +; the camera sits
about 11.6 deg below the body axis, measured against the tag-derived pitch on session 201826).
"""
import argparse
import csv
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from localization_pipeline import DEFAULT_REPO, load_repo, read_frames_csv, view_bounds  # noqa: E402

CAM_PITCH_OFFSET = -11.6          # deg: camera pitch = IMU pitch + this
FPS = 30.0
ANNOT = "viewer_video.mp4"


def to_plotly(p, floor_y):
    """Map frame (x right, y up, z out of wall) -> plot frame (X right, Y out of the wall, Z up from the floor)."""
    return [float(p[0]), float(p[2]), float(p[1]) - floor_y]


def transcode(src, dst, force):
    if dst.exists() and not force:
        return
    print(f"  transcoding {src.name} -> {dst.name} (H.264) ...", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=640:480", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-an", "-r", "30", str(dst)], check=True)


def build_session(sess, R, cfg, tag_map, floor_y, hfov, vfov, force):
    import plotly.graph_objects as go

    loc_path = sess / "localization" / "localization.csv"
    if not loc_path.exists():
        sys.exit(f"{sess}: run localization_pipeline.py first (no {loc_path})")
    frames = {r["frame_id"]: r for r in read_frames_csv(sess)}
    with open(loc_path) as f:
        loc = {r["frame_id"]: r for r in csv.DictReader(f)}
    n = len(frames)
    if len(loc) < 0.9 * n:
        print(f"  WARNING: localization.csv has {len(loc)} rows but the video has {n} frames "
              f"(was it a partial run?). Re-run localization_pipeline.py on this session.")

    # per-video-frame arrays (frame i <-> frame_id i+1)
    X, Y, Z, LOST, HEAD, PIT, USED, T = [], [], [], [], [], [], [], []
    last_h, last_row = 0.0, None
    for i in range(n):
        fid = str(i + 1)
        r = loc.get(fid, last_row)
        last_row = r
        T.append(round(float(r["t_rel"]), 2) if r else 0)
        if r and r["x"] != "":
            px, py, pz = to_plotly([float(r["x"]), float(r["y"]), float(r["z"])], floor_y)
            X.append(round(px, 3)); Y.append(round(py, 3)); Z.append(round(pz, 3))
        else:
            X.append(None); Y.append(None); Z.append(None)
        LOST.append(int(r["lost"]) if r and r["lost"] != "" else 1)
        if r and r["heading_deg"] != "":
            last_h = float(r["heading_deg"])
        HEAD.append(round(last_h, 1))
        imu = frames.get(fid)
        PIT.append(round((float(imu["pitch"]) if imu and imu["pitch"] != "" else 0.0) + CAM_PITCH_OFFSET, 1))
        USED.append([int(t) for t in r["tags_used"].split()] if r and r["tags_used"] else [])

    # --- static scene
    fig = go.Figure()
    pts = [[x, y, z] for x, y, z in zip(X, Y, Z) if x is not None]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    tag_pos = {int(t): to_plotly(M[:3, 3], floor_y) for t, M in tag_map.poses.items()}
    xr = (min([-0.6] + [np.percentile(xs, 1)] + [p[0] for p in tag_pos.values()]) - 0.4,
          max([0.6] + [np.percentile(xs, 99)] + [p[0] for p in tag_pos.values()]) + 0.4) if xs else (-1.5, 3.5)
    yr = (min([-0.5] + ([np.percentile(ys, 1)] if ys else []) + [p[1] for p in tag_pos.values()]),
          max([3.0] + ([np.percentile(ys, 99)] if ys else []) + [p[1] for p in tag_pos.values()]) + 0.3)
    zr = (0.0, max(2.4, (np.percentile(zs, 99) if zs else 2.0) + 0.3))
    fig.add_trace(go.Mesh3d(x=[xr[0], xr[1], xr[1], xr[0]], y=[yr[0], yr[0], yr[1], yr[1]], z=[0] * 4, i=[0, 0], j=[1, 2],
                            k=[2, 3], color="#8a7a66", opacity=0.25, name="floor", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Mesh3d(x=[xr[0], xr[1], xr[1], xr[0]], y=[0] * 4, z=[0, 0, zr[1], zr[1]], i=[0, 0], j=[1, 2],
                            k=[2, 3], color="#9aa7b5", opacity=0.18, name="wall (tag-10 wall)", hoverinfo="skip",
                            showlegend=False))
    col = {"origin": "#ff3b3b", "wall": "#3b82f6", "floor": "#ff9f1a"}
    labels_x, labels_y, labels_z, labels_t = [], [], [], []
    for tid, M in sorted(tag_map.poses.items()):
        size = (cfg.sizes_mm.get(tid) or 150.0) / 1000.0
        a = size / 2
        ux, uy, nrm = M[:3, 0], M[:3, 1], M[:3, 2]
        kind = "origin" if tid == cfg.origin_tag else ("floor" if abs(nrm[1]) > 0.7 else "wall")
        corners = [to_plotly(M[:3, 3] + sx * a * ux + sy * a * uy, floor_y)
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        fig.add_trace(go.Mesh3d(x=[c[0] for c in corners], y=[c[1] for c in corners], z=[c[2] for c in corners],
                                i=[0, 0], j=[1, 2], k=[2, 3], color=col[kind], opacity=0.95, name=f"tag {tid}",
                                hovertemplate=f"tag {tid} ({kind})<br>height %{{z:.2f}} m<extra></extra>",
                                showlegend=False))
        cx, cy, cz = tag_pos[int(tid)]
        labels_x.append(cx); labels_y.append(cy); labels_z.append(cz + 0.12 + size / 2)
        labels_t.append(f"{tid}" + (" (origin)" if kind == "origin" else "") + f"  {cz:.2f} m")
    fig.add_trace(go.Scatter3d(x=labels_x, y=labels_y, z=labels_z, mode="text", text=labels_t,
                               textfont=dict(color="white", size=13), hoverinfo="skip", showlegend=False))
    path_idx = len(fig.data)
    fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(width=2.5, color=list(range(n)),
                                                                     colorscale="Viridis"),
                               opacity=0.55, text=[f"t {t:.1f}s" for t in T], hoverinfo="text", showlegend=False,
                               name="path"))
    dyn = {}

    def add_dynamic(name, **kw):
        dyn[name] = len(fig.data)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], hoverinfo="skip", showlegend=False, name=name, **kw))

    add_dynamic("fix", mode="markers", marker=dict(size=7, color="#32e66e", line=dict(width=2, color="white")))
    add_dynamic("lost", mode="markers", marker=dict(size=7, color="#e63232", symbol="diamond-open",
                                                    line=dict(width=3, color="#e63232")))
    add_dynamic("trail", mode="lines", line=dict(width=6, color="#ffe14d"))
    add_dynamic("cone", mode="lines", line=dict(width=3, color="#7dffb0"))
    add_dynamic("taglines", mode="lines", line=dict(width=2, color="#9fe8a8", dash="dot"))
    add_dynamic("drop", mode="lines", line=dict(width=2, color="#cccccc", dash="dash"))

    ticks_y = list(range(int(np.ceil(yr[0])), int(np.floor(yr[1])) + 1))
    fig.update_layout(
        paper_bgcolor="#14161a", font=dict(color="#dddddd"), margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
        scene=dict(
            xaxis=dict(title="x: right of tag 10 (m)", range=list(xr), backgroundcolor="#1b1e24", gridcolor="#3a3f48"),
            yaxis=dict(title="Y: out from wall (m); toward origin = -Y", range=list(yr), tickvals=ticks_y,
                       ticktext=[str(t) for t in ticks_y], backgroundcolor="#1b1e24", gridcolor="#3a3f48"),
            zaxis=dict(title="height above floor (m)", range=list(zr), backgroundcolor="#1b1e24", gridcolor="#3a3f48"),
            aspectmode="data", camera=dict(eye=dict(x=1.0, y=1.7, z=0.8))))
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="scene", default_height="100%",
                            default_width="100%", config={"responsive": True, "displaylogo": False})
    data = dict(n=n, fps=FPS, X=X, Y=Y, Z=Z, LOST=LOST, HEAD=HEAD, PIT=PIT, USED=USED, T=T, tags=tag_pos,
                tH=float(np.tan(np.radians(hfov / 2))), tV=float(np.tan(np.radians(vfov / 2))),
                idx=dict(path=path_idx, **dyn))
    return plot_html, data


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>3D localization viewer - __NAME__</title>
<style>html,body{margin:0;height:100%;background:#14161a;color:#ddd;font:13px system-ui,sans-serif}
#wrap{display:flex;height:100vh}#left{width:34%;min-width:380px;padding:10px;box-sizing:border-box;overflow:auto}
#right{flex:1;min-width:0}video{width:100%;border-radius:6px;background:#000}
#info{font:13px ui-monospace,monospace;line-height:1.6;margin:10px 0;padding:8px 10px;background:#1b1e24;border-radius:6px}
.lost{color:#ff6b6b}.fix{color:#5be58a}small{color:#8b93a1}h3{margin:4px 0 8px;font-size:15px}</style></head><body>
<div id="wrap"><div id="left"><h3>__NAME__</h3>
<video id="v" src="__VIDEO__" controls muted playsinline></video>
<div id="info">loading...</div>
<small>Play the video or drag its timeline: the drone moves in the 3D scene. Click the coloured path to jump the video.
Green = tag fix; red = lost (holding last position). Cone = camera view; dotted lines = tags it is using.
Coordinates: +X right of tag 10, +Y out into the room, +Z up. Takeoff increases Z; approaching the origin decreases Y.
Drag to rotate the scene, scroll to zoom.</small></div>
<div id="right">__PLOT__</div></div>
<script>
const D = __DATA__;
const gd = document.getElementById('scene'), v = document.getElementById('v'), info = document.getElementById('info');
const IDX = D.idx, TRAIL = 120, DEG = Math.PI / 180;
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function update(i) {
  const x = D.X[i], y = D.Y[i], z = D.Z[i];
  const empty = {x: [[],[],[],[],[],[]], y: [[],[],[],[],[],[]], z: [[],[],[],[],[],[]]};
  const order = [IDX.fix, IDX.lost, IDX.trail, IDX.cone, IDX.taglines, IDX.drop];
  const lost = D.LOST[i] === 1;
  if (x === null) {
    Plotly.restyle(gd, empty, order);
    info.innerHTML = 't ' + D.T[i].toFixed(1) + ' s<br><span class="lost">no position yet (no mapped tag seen)</span>';
    return;
  }
  const h = D.HEAD[i] * DEG, p = D.PIT[i] * DEG;
  const f = [Math.cos(p)*Math.sin(h), -Math.cos(p)*Math.cos(h), Math.sin(p)];
  const r = [Math.cos(h), Math.sin(h), 0];
  const u = cross(f, r), d = 1.0;
  const corner = (sh, sv) => [x + d*(f[0] + sh*D.tH*r[0] + sv*D.tV*u[0]),
                              y + d*(f[1] + sh*D.tH*r[1] + sv*D.tV*u[1]),
                              z + d*(f[2] + sh*D.tH*r[2] + sv*D.tV*u[2])];
  const c = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)], a = [x, y, z], N = null;
  const seq = [a, c[0], a, c[1], a, c[2], a, c[3], N, c[0], c[1], c[2], c[3], c[0]];
  const cone = [0, 1, 2].map(k => seq.map(q => q === null ? null : q[k]));
  const tl = [[], [], []];
  for (const t of D.USED[i]) { const q = D.tags[t]; if (!q) continue;
    for (let k = 0; k < 3; k++) tl[k].push(a[k], q[k], null); }
  const tx = [], ty = [], tz = [];
  for (let j = Math.max(0, i - TRAIL); j <= i; j++) if (D.X[j] !== null) { tx.push(D.X[j]); ty.push(D.Y[j]); tz.push(D.Z[j]); }
  const pt = lost ? [[],[]] : [[x],[y]];
  Plotly.restyle(gd, {
    x: [lost ? [] : [x], lost ? [x] : [], tx, cone[0], tl[0], [x, x]],
    y: [lost ? [] : [y], lost ? [y] : [], ty, cone[1], tl[1], [y, y]],
    z: [lost ? [] : [z], lost ? [z] : [], tz, cone[2], tl[2], [z, 0]]}, order);
  info.innerHTML = 't ' + D.T[i].toFixed(1) + ' s &nbsp; frame ' + (i+1) + '/' + D.n + '<br>' +
    'x ' + x.toFixed(2) + ' &nbsp; Y (out from wall) ' + y.toFixed(2) + ' m<br>' +
    'height ' + z.toFixed(2) + ' m &nbsp; heading ' + D.HEAD[i].toFixed(0) + '&deg; &nbsp; cam pitch ' + D.PIT[i].toFixed(0) + '&deg;<br>' +
    (lost ? '<span class="lost">LOST - holding last position</span>' : '<span class="fix">fix via tags [' + D.USED[i].join(', ') + ']</span>');
}
let last = -1, lastT = 0;
function tick() {
  const i = Math.max(0, Math.min(D.n - 1, Math.round(v.currentTime * D.fps)));
  const now = performance.now();
  if (i !== last && now - lastT > 45) { update(i); last = i; lastT = now; }
  requestAnimationFrame(tick);
}
gd.on('plotly_click', e => { const pt = e.points && e.points[0];
  if (pt && pt.curveNumber === IDX.path) v.currentTime = pt.pointNumber / D.fps; });
v.addEventListener('loadedmetadata', () => { const m = /t=([\\d.]+)/.exec(location.hash); if (m) v.currentTime = +m[1]; });
update(0); tick();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--map", type=Path, default=HERE / "maps" / "latest_layout.yaml")
    ap.add_argument("--drone", default="tello_3")
    ap.add_argument("--floor-y", type=float, default=-1.22)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-transcode the video")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = ap.parse_args()

    R = load_repo(args.repo)
    cfg = R["load_tag_config"]()
    cam = R["CONFIG_DIR"] / "camera" / f"{args.drone}.yaml"
    K, _ = R["camera_from_dict"](R["load_yaml"](cam if cam.exists() else R["CONFIG_DIR"] / "camera" / "default.yaml"))
    hfov = float(np.degrees(2 * np.arctan(480.0 / K[0, 0])))
    vfov = float(np.degrees(2 * np.arctan(360.0 / K[1, 1])))
    tag_map = R["TagMap"].load(args.map)
    for sess in args.sessions:
        print(f"== {sess.name}")
        out = sess / "localization"
        transcode(sess / "video.mp4", out / ANNOT, args.force)
        plot_html, data = build_session(sess, R, cfg, tag_map, args.floor_y, hfov, vfov, args.force)
        html = (TEMPLATE.replace("__NAME__", sess.name).replace("__VIDEO__", ANNOT)
                .replace("__PLOT__", plot_html).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
        page = out / "viewer_3d.html"
        page.write_text(html)
        print(f"  wrote {page} ({page.stat().st_size / 1e6:.1f} MB)")
        if not args.no_open:
            webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
