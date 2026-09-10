"""SonarVision log (XTF) ingestion helpers.

Parses .xtf sonar log files, renders them to waterfall images, tiles them for
the detector, and merges detections across tile seams. Kept free of any model
dependency so it can be imported by notebooks, scripts, or the backend service.
"""

from pathlib import Path

import numpy as np

_CHANNEL_TYPE_PORT = 1
_CHANNEL_TYPE_STBD = 2


def _utc(dt64):
    """Convert a numpy.datetime64 to fractional UTC seconds (float)."""
    return (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")


def to_u16(arr):
    """Normalise raw XTF samples (any dtype) to a 16-bit unsigned image array."""
    arr = np.asarray(arr)
    dt = arr.dtype
    if dt == np.uint16:
        return arr
    if dt == np.uint8:
        return arr.astype(np.uint16) * 257
    if dt == np.int16:
        return (arr.astype(np.int32) + 32768).astype(np.uint16)
    if dt == np.int8:
        return (arr.astype(np.int16) + 128).astype(np.uint16) * 257
    if np.issubdtype(dt, np.floating):
        f = arr.astype(np.float32)
        mn = float(f.min())
        mx = float(f.max())
        if mn >= 0.0 and mx <= 1.0:
            return (np.clip(f, 0.0, 1.0) * 65535.0).astype(np.uint16)
        half = max(abs(mx), abs(mn), 1e-9)
        return (np.clip((f / half + 1.0) / 2.0, 0.0, 1.0) * 65535.0).astype(np.uint16)
    raise ValueError("unsupported XTF sample dtype: {}".format(dt))


def stack_channel(pings, channel):
    """Stack the per-channel samples of consecutive pings into a dense 2D array.

    Rows are in the same order as ``pings`` - callers sort pings by time first.
    Short pings are right-aligned (padded at near-nadir) so rows share a width.
    """
    rows = [np.asarray(p.data[channel]) for p in pings]
    sizes = [r.shape[0] for r in rows]
    if not sizes:
        return np.zeros((0, 0), dtype=np.uint16)
    max_sz = max(sizes)
    out = np.zeros((len(rows), max_sz), dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        sz = r.shape[0]
        if sz == max_sz:
            out[i] = r
        elif sz > 0:
            out[i, max_sz - sz:] = r
    return out


def extract_survey(xfile):
    """Extract sonar channels + navigation from a single .xtf file.

    Returns a dict:
      name          survey name (file stem)
      file          Path to the source log
      channels      {label: {"side": ..., "array": uint16[pings, samples],
                             "freq_hz": ...}}
      ping_time     np.ndarray of UTC seconds per ping row
      lat, lon      np.ndarray per ping row (or per-nav when grid mode)
      grid_is_latlon  True if lat/lon are real degrees (NavUnits=latlon)
    """
    from pyxtf import XTFChannelType, XTFHeaderType, XTFNavUnits, xtf_read

    xfile = Path(xfile)
    file_header, packets = xtf_read(str(xfile))
    pings = list(packets.get(XTFHeaderType.sonar, []))

    if not pings:
        raise RuntimeError("no sonar ping data in {}".format(xfile.name))

    # Sort pings by time so strip rows are chronological
    pings.sort(key=lambda p: p.get_time())

    # Determine which per-ping channel indices are sidescan (port/starboard)
    sonar_nums = {c.ChannelNumber for c in file_header.sonar_info}
    chan_indices = [
        i for i, h in enumerate(pings[0].ping_chan_headers)
        if h.ChannelNumber in sonar_nums
    ]
    if not chan_indices:
        chan_indices = list(range(len(pings[0].ping_chan_headers)))

    def side_of(channel_number):
        for c in file_header.ChanInfo:
            if c.ChannelNumber == channel_number:
                if c.TypeOfChannel == XTFChannelType.port:
                    return "port"
                if c.TypeOfChannel == XTFChannelType.stbd:
                    return "stbd"
                return "sonar"
        return "sonar"

    channels = {}
    for i in chan_indices:
        hdr = pings[0].ping_chan_headers[i]
        arr = stack_channel(pings, i)
        if arr.shape[1] == 0:
            continue
        channels["ch{}".format(i)] = {
            "side": side_of(hdr.ChannelNumber),
            "array": to_u16(arr),
            "freq_hz": float(hdr.Frequency),
        }

    if not channels:
        raise RuntimeError("no usable sidescan samples in {}".format(xfile.name))

    # Navigation
    fixes = []
    for htype in (XTFHeaderType.navigation, XTFHeaderType.pos_raw_navigation):
        for p in packets.get(htype, []):
            try:
                t = float(_utc(p.get_time()))
            except Exception:
                continue
            fixes.append((t, float(p.RawYcoordinate), float(p.RawXcoordinate)))
    fixes.sort(key=lambda r: r[0])

    grid_is_latlon = int(file_header.NavUnits) == int(XTFNavUnits.latlon)
    n_pings = len(pings)
    lat = np.full(n_pings, np.nan)
    lon = np.full(n_pings, np.nan)
    ping_time = np.array([float(_utc(p.get_time())) for p in pings])

    if len(fixes) >= 2:
        nt = np.array([r[0] for r in fixes])
        nlat = np.array([r[1] for r in fixes])
        nlon = np.array([r[2] for r in fixes])
        lat = np.interp(ping_time, nt, nlat)
        lon = np.interp(ping_time, nt, nlon)

    return {
        "name": xfile.stem,
        "file": xfile,
        "channels": channels,
        "ping_time": ping_time,
        "lat": lat,
        "lon": lon,
        "grid_is_latlon": grid_is_latlon,
    }


def render_waterfall(u16, use_db=True, stretch_percentile=1.0):
    """Turn a uint16 waterfall strip into an 8-bit grayscale visualization."""
    f = u16.astype(np.float32)
    f = np.clip(f, 1.0, None)
    if use_db:
        f = 20.0 * np.log10(f)
    lo = np.percentile(f, stretch_percentile)
    hi = np.percentile(f, 100.0 - stretch_percentile)
    img = np.clip((f - lo) * (255.0 / max(float(hi - lo), 1.0)), 0.0, 255.0)
    return img.astype(np.uint8)


def make_tiles(img, tile_size, overlap):
    """Slice a strip into overlapping square tiles.

    Yields dicts: {"row0", "row1", "tile"} where row0/row1 are the strip row
    range covered by the tile (the mapping needed for geotagging).
    """
    h, w = img.shape
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    step = max(tile_size - overlap, 1)
    out = []
    for r0 in range(0, max(h, 1), step):
        r1 = min(r0 + tile_size, h)
        tile = img[r0:r1]
        if tile.shape[0] < tile_size:
            pad = tile_size - tile.shape[0]
            tile = np.pad(tile, ((0, pad), (0, 0)), mode="edge")
        out.append({"row0": r0, "row1": r1, "tile": tile})
    return out


def merge_detections(dets, iou_thr=0.35, center_tol=64):
    """Merge detections that repeat the same object across overlapping tiles.

    dets: list of dicts with keys x1,y1,x2,y2,score,cls,row_start,tile,channel.
    Boxes are expressed in strip (survey) row coordinates. Returns merged list.
    """
    kept = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        dup = None
        for k in kept:
            if d["cls"] != k["cls"]:
                continue
            # IoU in strip coordinates
            xa, ya, xb, yb = d["x1"], d["y1"], d["x2"], d["y2"]
            xc, yc, xd, yd = k["x1"], k["y1"], k["x2"], k["y2"]
            ix1, iy1 = max(xa, xc), max(ya, yc)
            ix2, iy2 = min(xb, xd), min(yb, yd)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (xb - xa) * (yb - ya) + (xd - xc) * (yd - yc) - inter
            iou = inter / max(union, 1e-9)
            cdist = abs(((ya + yb) / 2.0) - ((yc + yd) / 2.0))
            if iou >= iou_thr or cdist <= center_tol:
                dup = k
                break
        if dup is not None:
            # extend the box to cover both + keep the strongest score
            dup["x1"] = min(dup["x1"], d["x1"])
            dup["y1"] = min(dup["y1"], d["y1"])
            dup["x2"] = max(dup["x2"], d["x2"])
            dup["y2"] = max(dup["y2"], d["y2"])
        else:
            kept.append(dict(d))
    return kept