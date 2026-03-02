"""
Build Me Money AI — RunPod Serverless Handler
Two output paths:
  - "short"   → 1080x1920 portrait, fill-screen talking head, bg music, no graphics
  - "youtube" → 1920x1080 landscape, dynamic cuts (face ↔ B-roll), FFmpeg-generated
                 animated graphics (intro banner, subscribe, like, outro), bg music
                 #v2
"""

import runpod
import os
import subprocess
import requests
import boto3
import tempfile
import json
import math
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
R2_BUCKET        = "buildmemoney-videos"
R2_ENDPOINT      = os.environ.get("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY    = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY    = os.environ.get("R2_SECRET_ACCESS_KEY", "")
BG_MUSIC_URL     = os.environ.get(
    "BG_MUSIC_URL",
    "https://pub-223fe8c1dd984b078073889c196fed45.r2.dev/bg-music.mp3"
)
FONT_PATH        = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_ITALIC_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"
PEXELS_API_KEY   = os.environ.get("PEXELS_API_KEY", "")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")

# ── Model bootstrap ───────────────────────────────────────────────────────────

def ensure_models():
    """
    Download SadTalker + GFPGAN model weights on first startup if not present.
    Uses HF_TOKEN env var for HuggingFace authentication.
    Models are saved to /app/SadTalker/checkpoints and /app/SadTalker/gfpgan/weights.
    """
    sadtalker_dir = Path("/app/SadTalker")
    ckpt_dir      = sadtalker_dir / "checkpoints"
    gfp_dir       = sadtalker_dir / "gfpgan" / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gfp_dir.mkdir(parents=True, exist_ok=True)

   # SadTalker checkpoints from GitHub releases (stable, no auth needed)
    sadtalker_files = {
        "SadTalker_V0.0.2_256.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
        "mapping_00109-model.pth.tar":       "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar":       "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
    }
    for fname, url in sadtalker_files.items():
        dest = ckpt_dir / fname
        if not dest.exists():
            print(f"[MODELS] Downloading {fname} from GitHub releases...")
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)

    # GFPGAN / facexlib weights from GitHub releases (no auth needed)
    github_weights = [
        ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
         gfp_dir / "GFPGANv1.4.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
         gfp_dir / "detection_Resnet50_Final.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
         gfp_dir / "parsing_parsenet.pth"),
    ]
    for url, dest in github_weights:
        if not dest.exists():
            print(f"[MODELS] Downloading {dest.name}...")
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)

    print("[MODELS] All model weights present.")

# Run model download at startup (before RunPod starts accepting jobs)
ensure_models()

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a shell command, raise on failure."""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    return result

def probe_duration(path):
    """Return video duration in seconds using ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", path],
        capture_output=True, text=True, check=True
    )
    info = json.loads(r.stdout)
    return float(info["format"]["duration"])

def download_file(url, dest):
    """Download a URL to a local path."""
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return dest

def upload_to_r2(local_path, key):
    """Upload file to Cloudflare R2, return public URL."""
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    s3.upload_file(local_path, R2_BUCKET, key,
                   ExtraArgs={"ContentType": "video/mp4"})
    public_base = os.environ.get(
        "R2_PUBLIC_URL",
        "https://pub-223fe8c1dd984b078073889c196fed45.r2.dev"
    )
    return f"{public_base}/{key}"

def fetch_broll_from_pexels(niche, count=6):
    """Auto-fetch B-roll video URLs from Pexels for a given niche keyword."""
    if not PEXELS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": niche, "per_page": count, "orientation": "landscape"},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        urls = []
        for vid in data.get("videos", []):
            # Prefer 1080p, fall back to highest available
            files = sorted(vid.get("video_files", []),
                           key=lambda f: f.get("height", 0), reverse=True)
            for f in files:
                if f.get("height", 0) >= 720:
                    urls.append(f["link"])
                    break
        return urls
    except Exception as e:
        print(f"[PEXELS] Warning: {e}")
        return []

# ── SadTalker ─────────────────────────────────────────────────────────────────

def run_sadtalker(source_image, driven_audio, output_dir,
                  preprocess="full", still=False, use_enhancer=False, size=256):
    """Run the SadTalker inference script."""
    cmd = [
        "python", "inference.py",
        "--source_image", source_image,
        "--driven_audio", driven_audio,
        "--result_dir", output_dir,
        "--preprocess", preprocess,
        "--size", str(size),
        "--face3dvis",
    ]
    if still:
        cmd.append("--still")
    if use_enhancer:
        cmd += ["--enhancer", "gfpgan"]

    cwd = "/app/SadTalker"
    if not os.path.isdir(cwd):
        cwd = os.getcwd()  # fallback for testing

    subprocess.run(cmd, cwd=cwd, check=True)

    # SadTalker writes to output_dir/<source_name>_<timestamp>.mp4
    results = list(Path(output_dir).glob("*.mp4"))
    if not results:
        raise RuntimeError("SadTalker produced no output video.")
    return str(sorted(results)[-1])

# ── Alpha expression generator ────────────────────────────────────────────────

def alpha_expr(t_in, t_out, fade=0.4):
    """
    Return an FFmpeg expression string for a smooth fade-in / hold / fade-out alpha.
    The expression evaluates to [0, 1] and is safe to embed inside single-quotes
    in an FFmpeg filter_complex string.
    """
    ie = t_in + fade
    oe = t_out + fade
    return (
        f"if(lt(t,{t_in:.2f}),0,"
        f"if(lt(t,{ie:.2f}),(t-{t_in:.2f})/{fade:.1f},"
        f"if(lt(t,{t_out:.2f}),1,"
        f"if(lt(t,{oe:.2f}),({oe:.2f}-t)/{fade:.1f},0))))"
    )

# ── Cut schedule ──────────────────────────────────────────────────────────────

def make_cut_schedule(duration, n_broll):
    """
    Build a list of scene cuts for the YouTube long-form video.
    Pattern: open face 5s → alternate B-roll 12s / face 7s → close with face.
    Each entry: {type, start, end[, broll_idx]}
    """
    if n_broll == 0:
        return [{"type": "face", "start": 0.0, "end": duration}]

    cuts = [{"type": "face", "start": 0.0, "end": min(5.0, duration)}]
    t = 5.0
    br = 0

    while t < duration - 5:
        # B-roll segment
        broll_dur = min(12.0, duration - t - 5.0)
        if broll_dur < 3.0:
            break
        cuts.append({
            "type": "broll",
            "start": t,
            "end": t + broll_dur,
            "broll_idx": br % n_broll
        })
        t += broll_dur
        br += 1

        if t >= duration - 5:
            break

        # Face segment
        face_dur = min(7.0, duration - t - 5.0)
        if face_dur < 2.0:
            break
        cuts.append({"type": "face", "start": t, "end": t + face_dur})
        t += face_dur

    # Closing face segment (always end on face)
    last_end = cuts[-1]["end"] if cuts else 0.0
    if last_end < duration - 0.1:
        cuts.append({"type": "face", "start": last_end, "end": duration})

    return cuts

# ── SHORT path ────────────────────────────────────────────────────────────────

def produce_short(raw_video, bg_music, output_path):
    """
    Portrait 1080×1920. Talking head fills screen (blurred background behind
    square face to eliminate black bars). Low-volume bg music. No graphics.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", bg_music,
        "-filter_complex",
        # Face fills full width (1080), blurred bg fills full 1920 height
        "[0:v]scale=1080:1080,setsar=1[face];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=28:6,setsar=1[bg];"
        "[bg][face]overlay=x=(W-w)/2:y=(H-h)/2[vout];"
        "[0:a]volume=1.0[voice];"
        "[1:a]volume=0.10[music];"
        "[voice][music]amix=inputs=2:duration=first:dropout_transition=2[audio]",
        "-map", "[vout]",
        "-map", "[audio]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    run(cmd)

# ── YOUTUBE path ──────────────────────────────────────────────────────────────

def _preprocess_face_landscape(raw_video, tmp_dir):
    """
    Scale the SadTalker square output to 1920×1080 landscape with blurred BG
    (no black bars). Returns path to pre-processed face video.
    """
    out = os.path.join(tmp_dir, "face_landscape.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video,
        "-filter_complex",
        # Face: scale to fit 1920×1080 height (1080×1080), centered
        "[0:v]scale=1080:1080,setsar=1[face];"
        # BG: scale+crop 1920×1080 + blur
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,boxblur=24:5,setsar=1[bg];"
        "[bg][face]overlay=x=(W-w)/2:y=0[vout]",
        "-map", "[vout]",
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        out
    ]
    run(cmd)
    return out

def _preprocess_broll(broll_path, duration, tmp_dir, idx):
    """Loop and scale a B-roll clip to 1920×1080 with enough duration."""
    out = os.path.join(tmp_dir, f"broll_{idx}.mp4")
    target_dur = duration + 5  # a bit extra to be safe
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", broll_path,
        "-t", str(target_dur),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,"
               "crop=1920:1080,setsar=1",
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        out
    ]
    run(cmd)
    return out

def _extract_segment(source_video, start, dur, tmp_dir, seg_idx):
    """Fast stream-copy extract of a segment from a pre-processed video."""
    out = os.path.join(tmp_dir, f"seg_{seg_idx:03d}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(dur),
        "-i", source_video,
        "-c", "copy",
        out
    ]
    run(cmd)
    return out

def _concat_segments(segment_paths, tmp_dir):
    """Concatenate a list of MP4 segments using FFmpeg concat demuxer."""
    list_file = os.path.join(tmp_dir, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    out = os.path.join(tmp_dir, "concat_raw.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out
    ]
    run(cmd)
    return out

def _apply_overlays_and_audio(concat_video, raw_video, bg_music, duration, output_path):
    """
    Final FFmpeg pass: mix audio and render all animated graphics via drawtext/drawbox.
    All animations use alpha expressions — no PNG files required.
    Graphics:
      • Intro banner   (0–3.5s):   Black bar + "BUILD ME MONEY AI" + tagline
      • Subscribe CTA  (8–14s):    Red box + "▶ SUBSCRIBE" + bell icon text
      • Comment CTA    (~35%):     Center bottom box + "💬 Drop a comment below!"
      • Like prompt    (~55%):     Left side box + "👍 Smash Like!"
      • Share CTA      (~68%):     Top-right box + "🔗 Share this video!"
      • Outro          (last 8s):  Bottom banner + 3 lines of CTA text
    """
    d = duration
    outro_start = max(d - 8.0, d * 0.80)

    # Alpha expressions
    a_intro      = alpha_expr(0.0,  3.5,  0.35)
    a_sub        = alpha_expr(8.0,  14.0, 0.40)
    a_like_start = min(22.0, d * 0.55)
    a_like       = alpha_expr(a_like_start, a_like_start + 6.0, 0.40)
    a_comment_start = min(d * 0.35, outro_start - 10.0)
    a_comment_start = max(a_comment_start, 16.0)          # never overlap subscribe
    a_comment    = alpha_expr(a_comment_start, a_comment_start + 6.0, 0.40)
    a_share_start = min(d * 0.68, outro_start - 8.0)
    a_share_start = max(a_share_start, a_like_start + 8.0) # never overlap like
    a_share      = alpha_expr(a_share_start, a_share_start + 5.0, 0.40)
    a_outro      = alpha_expr(outro_start, d, 0.50)

    # Pre-build font references for cleanliness
    font      = FONT_PATH
    font_en   = FONT_ITALIC_PATH

    # ── filter_complex ──────────────────────────────────────────────────────
    # NOTE: All alpha= values use single-quoted FFmpeg expressions.
    # All drawtext/drawbox filters are chained from [v0] → [vout].
    fc = (
        # ── Inputs ──
        # [v0] = concat video (video only)
        # [raw] = SadTalker output (audio only)
        # [bgm] = background music
        "[0:v]setsar=1[v0];"

        # ── INTRO BANNER ── (0–3.5s) ──────────────────────────────────────
        # Dark semi-transparent bar across top
        f"[v0]drawbox="
            f"x=0:y=0:w=iw:h=120:"
            f"color=black@0.75:t=fill:"
            f"enable='between(t,0,3.9)'[v1];"

        # Brand name
        f"[v1]drawtext="
            f"fontfile={font}:"
            f"text='BUILD ME MONEY AI':"
            f"fontsize=52:fontcolor=gold:"
            f"x=(w-text_w)/2:y=18:"
            f"alpha='{a_intro}'[v2];"

        # Tagline
        f"[v2]drawtext="
            f"fontfile={font_en}:"
            f"text='Your Daily Financial Freedom Content':"
            f"fontsize=26:fontcolor=white:"
            f"x=(w-text_w)/2:y=76:"
            f"alpha='{a_intro}'[v3];"

        # ── SUBSCRIBE CTA ── (8–14s) ──────────────────────────────────────
        # Red box
        f"[v3]drawbox="
            f"x=iw-320:y=ih-160:w=300:h=110:"
            f"color=red@0.90:t=fill:"
            f"enable='between(t,7.6,14.4)'[v4];"

        # White border
        f"[v4]drawbox="
            f"x=iw-320:y=ih-160:w=300:h=110:"
            f"color=white@0.95:t=3:"
            f"enable='between(t,7.6,14.4)'[v5];"

        # SUBSCRIBE text
        f"[v5]drawtext="
            f"fontfile={font}:"
            f"text='▶  SUBSCRIBE':"
            f"fontsize=34:fontcolor=white:"
            f"x=iw-298:y=ih-146:"
            f"alpha='{a_sub}'[v6];"

        # Bell / notification line
        f"[v6]drawtext="
            f"fontfile={font}:"
            f"text='🔔 Turn on notifications':"
            f"fontsize=19:fontcolor=white:"
            f"x=iw-298:y=ih-105:"
            f"alpha='{a_sub}'[v7];"

        # ── LIKE PROMPT ── (appears ~55% through) ─────────────────────────
        # Box on left side
        f"[v7]drawbox="
            f"x=20:y=ih/2-60:w=260:h=100:"
            f"color=black@0.70:t=fill:"
            f"enable='between(t,{a_like_start-0.4:.2f},{a_like_start+6.4:.2f})'[v8];"

        # Like text
        f"[v8]drawtext="
            f"fontfile={font}:"
            f"text='👍 Smash Like!':"
            f"fontsize=28:fontcolor=white:"
            f"x=30:y=ih/2-40:"
            f"alpha='{a_like}'[v9];"

        f"[v9]drawtext="
            f"fontfile={font}:"
            f"text='It really helps us!':"
            f"fontsize=20:fontcolor=lightyellow:"
            f"x=30:y=ih/2+0:"
            f"alpha='{a_like}'[v10];"

        # ── COMMENT CTA ── (appears ~35% through) ──────────────────────────
        # Box bottom-center
        f"[v10]drawbox="
            f"x=(iw-360)/2:y=ih-200:w=360:h=90:"
            f"color=black@0.75:t=fill:"
            f"enable='between(t,{a_comment_start-0.4:.2f},{a_comment_start+6.4:.2f})'[v11];"

        # Comment icon + text
        f"[v11]drawtext="
            f"fontfile={font}:"
            f"text='💬 Drop a comment below!':"
            f"fontsize=26:fontcolor=white:"
            f"x=(w-text_w)/2:y=ih-188:"
            f"alpha='{a_comment}'[v12];"

        # Comment sub-line
        f"[v12]drawtext="
            f"fontfile={font_en}:"
            f"text='We read every single one':"
            f"fontsize=19:fontcolor=lightyellow:"
            f"x=(w-text_w)/2:y=ih-155:"
            f"alpha='{a_comment}'[v13];"

        # ── SHARE CTA ── (appears ~68% through) ────────────────────────────
        # Box top-right corner
        f"[v13]drawbox="
            f"x=iw-330:y=10:w=310:h=90:"
            f"color=black@0.75:t=fill:"
            f"enable='between(t,{a_share_start-0.4:.2f},{a_share_start+5.4:.2f})'[v14];"

        # Share icon + text
        f"[v14]drawtext="
            f"fontfile={font}:"
            f"text='🔗 Share this video!':"
            f"fontsize=26:fontcolor=white:"
            f"x=iw-318:y=18:"
            f"alpha='{a_share}'[v15];"

        # Share sub-line
        f"[v15]drawtext="
            f"fontfile={font_en}:"
            f"text='Help someone else win financially':"
            f"fontsize=17:fontcolor=lightyellow:"
            f"x=iw-318:y=52:"
            f"alpha='{a_share}'[v16];"

        # ── OUTRO BANNER ── (last 8s) ─────────────────────────────────────
        # Bottom dark bar
        f"[v16]drawbox="
            f"x=0:y=ih-170:w=iw:h=170:"
            f"color=black@0.80:t=fill:"
            f"enable='between(t,{outro_start-0.5:.2f},{d:.2f})'[v17];"

        # Line 1
        f"[v17]drawtext="
            f"fontfile={font}:"
            f"text='Thanks for watching — Follow for daily money content!':"
            f"fontsize=30:fontcolor=gold:"
            f"x=(w-text_w)/2:y=ih-155:"
            f"alpha='{a_outro}'[v18];"

        # Line 2
        f"[v18]drawtext="
            f"fontfile={font}:"
            f"text='🔔 Subscribe · 👍 Like · 💬 Comment · 🔗 Share':"
            f"fontsize=24:fontcolor=white:"
            f"x=(w-text_w)/2:y=ih-115:"
            f"alpha='{a_outro}'[v19];"

        # Line 3
        f"[v19]drawtext="
            f"fontfile={font}:"
            f"text='New video every day  →  Build Me Money AI':"
            f"fontsize=20:fontcolor=lightyellow:"
            f"x=(w-text_w)/2:y=ih-80:"
            f"alpha='{a_outro}'[vout];"

        # ── AUDIO MIX ────────────────────────────────────────────────────
        "[1:a]volume=1.0[voice];"
        "[2:a]volume=0.10[music];"
        "[voice][music]amix=inputs=2:duration=first:dropout_transition=2[audio]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", concat_video,      # 0: video
        "-i", raw_video,          # 1: original SadTalker audio (voice)
        "-i", bg_music,           # 2: background music
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", "[audio]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    run(cmd)

def produce_youtube(raw_video, broll_paths, bg_music, output_path):
    """
    Landscape 1920×1080 with dynamic scene cuts:
      face full-screen ↔ B-roll full-screen
    Animated FFmpeg graphics on top. No pre-made PNG files needed.
    """
    duration = probe_duration(raw_video)

    with tempfile.TemporaryDirectory(prefix="yt_") as tmp:
        # 1. Pre-process face to landscape (no black bars)
        print("[YOUTUBE] Pre-processing face to landscape…")
        face_landscape = _preprocess_face_landscape(raw_video, tmp)

        # 2. Pre-process B-roll clips
        print(f"[YOUTUBE] Pre-processing {len(broll_paths)} B-roll clip(s)…")
        broll_processed = []
        for i, bp in enumerate(broll_paths):
            broll_processed.append(_preprocess_broll(bp, duration, tmp, i))

        # 3. Build cut schedule
        n_broll = len(broll_processed)
        cuts = make_cut_schedule(duration, n_broll)
        print(f"[YOUTUBE] Cut schedule ({len(cuts)} segments):")
        for c in cuts:
            print(f"  {c}")

        # 4. Extract segments
        segments = []
        for i, cut in enumerate(cuts):
            seg_dur = cut["end"] - cut["start"]
            if cut["type"] == "face":
                src = face_landscape
            else:
                src = broll_processed[cut["broll_idx"]]
            seg = _extract_segment(src, cut["start"], seg_dur, tmp, i)
            segments.append(seg)

        # 5. Concatenate segments
        print("[YOUTUBE] Concatenating segments…")
        concat_video = _concat_segments(segments, tmp)

        # 6. Apply animated overlays and mix audio
        print("[YOUTUBE] Applying animated overlays and mixing audio…")
        _apply_overlays_and_audio(concat_video, raw_video, bg_music, duration, output_path)

    print(f"[YOUTUBE] Done → {output_path}")

# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(event):
    """
    RunPod serverless entry point.
    Expected input fields:
      source_image  : URL to avatar image (JPG/PNG)
      driven_audio  : URL to ElevenLabs MP3
      job_type      : "short" | "youtube"  (default: "short")
      broll_clips   : list of video URLs   (used for "youtube" path)
      niche         : keyword for Pexels auto-fetch if broll_clips is empty
      preprocess    : SadTalker preprocess mode (default: "full")
      still         : SadTalker still mode (default: False)
      use_enhancer  : use GFPGAN enhancer  (default: False)
      size          : SadTalker size       (default: 256)
      video_title   : used for R2 filename (default: "output")
    """
    inp = event.get("input", {})

    source_image_url = inp["source_image"]
    driven_audio_url = inp["driven_audio"]
    job_type         = inp.get("job_type", inp.get("type", "short")).lower()
    broll_clips      = inp.get("broll_clips", [])
    niche            = inp.get("niche", "finance")
    preprocess       = inp.get("preprocess", "full")
    still            = inp.get("still", False)
    use_enhancer     = inp.get("use_enhancer", False)
    size             = inp.get("size", 256)
    video_title      = inp.get("video_title", "output").replace(" ", "_")[:60]

    print(f"[HANDLER] job_type={job_type} | title={video_title}")

    with tempfile.TemporaryDirectory(prefix="bmm_") as work_dir:
        # ── Download inputs ──────────────────────────────────────────────
        print("[HANDLER] Downloading inputs…")
        avatar_path = os.path.join(work_dir, "avatar.jpg")
        audio_path  = os.path.join(work_dir, "audio.mp3")
        music_path  = os.path.join(work_dir, "bgmusic.mp3")

        # Use per-job bg_music_url if provided (set by n8n Pixabay node), else fall back to env var
        music_url = inp.get("bg_music_url") or BG_MUSIC_URL
        print(f"[HANDLER] Music URL: {music_url}")

        download_file(source_image_url, avatar_path)
        download_file(driven_audio_url, audio_path)
        download_file(music_url,        music_path)

        # ── SadTalker ────────────────────────────────────────────────────
        sadtalker_out = os.path.join(work_dir, "sadtalker_out")
        os.makedirs(sadtalker_out, exist_ok=True)
        print("[HANDLER] Running SadTalker…")
        raw_video = run_sadtalker(
            avatar_path, audio_path, sadtalker_out,
            preprocess=preprocess, still=still,
            use_enhancer=use_enhancer, size=size
        )
        print(f"[HANDLER] SadTalker output: {raw_video}")

        # ── Post-processing ───────────────────────────────────────────────
        output_filename = f"{video_title}_{job_type}.mp4"
        output_path     = os.path.join(work_dir, output_filename)

        if job_type == "youtube":
            # Fetch B-roll if not provided
            if not broll_clips and niche:
                print(f"[HANDLER] Fetching B-roll from Pexels for niche: {niche}")
                broll_clips = fetch_broll_from_pexels(niche, count=6)

            broll_paths = []
            for i, url in enumerate(broll_clips):
                bp = os.path.join(work_dir, f"broll_raw_{i}.mp4")
                download_file(url, bp)
                broll_paths.append(bp)

            produce_youtube(raw_video, broll_paths, music_path, output_path)

        else:  # "short" (default)
            print("[HANDLER] Producing SHORT (portrait) video…")
            produce_short(raw_video, music_path, output_path)

        # ── Upload to R2 ─────────────────────────────────────────────────
        r2_key  = f"videos/{output_filename}"
        print(f"[HANDLER] Uploading to R2: {r2_key}")
        video_url = upload_to_r2(output_path, r2_key)
        print(f"[HANDLER] Uploaded: {video_url}")

    return {
        "video_url":  video_url,
        "job_type":   job_type,
        "title":      inp.get("video_title", "output"),
        "r2_key":     r2_key,
    }

# ── Entrypoint ────────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
