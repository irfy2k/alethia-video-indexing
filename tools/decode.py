"""PyAV decode of mp4 bytes -> (t_rel_seconds, bgr ndarray), rotated upright."""
import io
import math

import av
import numpy as np


class DecodeError(Exception):
    pass


def get_rotation(stream, frame=None):
    """Display rotation in degrees, COUNTERCLOCKWISE (ffmpeg av_display_rotation_get convention:
    a phone video tagged 'rotate 90' reports -90, i.e. rotate 90 deg clockwise to display)."""
    rot = getattr(frame, "rotation", None) if frame is not None else None
    if rot is not None:
        return rot
    sources = []
    if frame is not None:
        sources.append(getattr(frame, "side_data", None))
    sources.append(getattr(stream, "side_data", None))
    for sd in sources:
        try:
            m = sd.get("DISPLAYMATRIX") if sd is not None else None
            if m is not None:
                m = np.frombuffer(bytes(m), dtype="<i4")[:9] / 65536.0
                s0, s1 = math.hypot(m[0], m[3]), math.hypot(m[1], m[4])
                return -math.degrees(math.atan2(m[1] / s1, m[0] / s0))
        except Exception:
            pass
    try:  # old ffmpeg exposed the tag as metadata
        return -float(stream.metadata.get("rotate", 0))
    except Exception:
        return 0.0


def rotate(img, deg):
    """np.rot90 rotates counterclockwise for k>0, matching the CCW-degrees convention above."""
    k = int(round(deg / 90.0)) % 4
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def decode(mp4_bytes):
    """Yield (t_rel, bgr). Raises DecodeError (after yielding what decoded) on failure."""
    try:
        if not mp4_bytes:
            raise ValueError("empty video")
        c = av.open(io.BytesIO(mp4_bytes))
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        rate = float(s.average_rate or s.guessed_rate or 25)
    except Exception as e:
        raise DecodeError(f"could not open video: {str(e)[:100]}") from e
    try:
        t0, rot = None, None
        for i, f in enumerate(c.decode(s)):
            t = f.time if f.pts is not None else i / rate
            if t0 is None:
                t0, rot = t, get_rotation(s, f)
            img = f.to_ndarray(format="bgr24")
            yield max(0.0, t - t0), (rotate(img, rot) if rot else img)
    except Exception as e:
        raise DecodeError(f"decode failed: {str(e)[:100]}") from e
    finally:
        c.close()
