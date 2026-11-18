"""Create a shareable visualization of a predicted route and optional reference."""
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd


def _route_coordinates(arc_ids: list[int], nodes: pd.DataFrame, arcs: pd.DataFrame) -> list[list[float]]:
    coords: list[list[float]] = []
    for arc_id in arc_ids:
        row = arcs.iloc[int(arc_id)]
        a, b = nodes.iloc[int(row.from_node_id)], nodes.iloc[int(row.to_node_id)]
        segment = [[float(a.latitude), float(a.longitude)], [float(b.latitude), float(b.longitude)]]
        if coords and coords[-1] == segment[0]:
            coords.append(segment[1])
        else:
            coords.extend(segment)
    return coords


def _save_svg_map(gps, nodes, arcs, predicted_arc_ids, output, reference_arc_ids=None):
    """Dependency-free route map fallback rendered as an embedded SVG."""
    lat0 = float(gps.latitude.mean())
    scale_x = max(0.15, math.cos(math.radians(lat0)))
    def coords_for(ids):
        values = []
        for arc_id in ids:
            arc = arcs.iloc[int(arc_id)]
            for node_id in (int(arc.from_node_id), int(arc.to_node_id)):
                node = nodes.iloc[node_id]
                values.append((float(node.longitude) * scale_x, float(node.latitude)))
        return values
    gps_xy = [(float(x) * scale_x, float(y)) for x, y in gps[["longitude", "latitude"]].itertuples(index=False, name=None)]
    predicted, reference = coords_for(predicted_arc_ids), coords_for(reference_arc_ids or [])
    all_xy = gps_xy + predicted + reference
    xmin, xmax = min(x for x, _ in all_xy), max(x for x, _ in all_xy)
    ymin, ymax = min(y for _, y in all_xy), max(y for _, y in all_xy)
    dx, dy = max(xmax - xmin, 1e-7), max(ymax - ymin, 1e-7)
    xmin, xmax = xmin - dx * .04, xmax + dx * .04
    ymin, ymax = ymin - dy * .04, ymax + dy * .04
    def point(x, y):
        return (f"{30 + 940 * (x - xmin) / (xmax - xmin):.2f},"
                f"{770 - 720 * (y - ymin) / (ymax - ymin):.2f}")
    def polyline(values):
        return " ".join(point(x, y) for x, y in values)
    ref_svg = (f'<polyline points="{polyline(reference)}" fill="none" stroke="#e53935" '
               'stroke-width="2" stroke-dasharray="7 6" opacity=".8"/>' if reference else "")
    start_x, start_y = point(*gps_xy[0]).split(",")
    end_x, end_y = point(*gps_xy[-1]).split(",")
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><title>PathSnap route</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;color:#17212b;background:#f4f7fb}}main{{max-width:1100px;margin:auto}}svg{{width:100%;height:auto;background:#fff;border:1px solid #d8e0e9;border-radius:12px}}.legend{{display:flex;gap:24px;margin:12px 0}}.item::before{{content:"";display:inline-block;width:22px;height:4px;margin:0 7px 3px 0;background:var(--c)}}.note{{color:#526273;font-size:13px}}</style></head>
<body><main><h1>PathSnap route visualization</h1><p class="note">GPS observations and road network paths shown in longitude/latitude space.</p>
<div class="legend"><span class="item" style="--c:#1565c0">Predicted route</span><span class="item" style="--c:#e53935">Reference route</span><span class="item" style="--c:#616161">GPS trace</span></div>
<svg viewBox="0 0 1000 820" role="img" aria-label="Predicted and reference route"><rect width="1000" height="820" fill="#fff"/>
<polyline points="{polyline(gps_xy)}" fill="none" stroke="#616161" stroke-width="1.5" opacity=".45"/>{ref_svg}
<polyline points="{polyline(predicted)}" fill="none" stroke="#1565c0" stroke-width="3.5" opacity=".92"/>
<circle cx="{start_x}" cy="{start_y}" r="6" fill="#2e7d32"/><circle cx="{end_x}" cy="{end_y}" r="6" fill="#6a1b9a"/>
<text x="30" y="805" fill="#526273" font-size="13">Start</text><text x="90" y="805" fill="#526273" font-size="13">Green: start · Purple: end</text></svg></main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")
    return output


def save_route_map(gps: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                   predicted_arc_ids: list[int], output_html: str | Path,
                   reference_arc_ids: list[int] | None = None) -> Path:
    """Save an interactive Folium map or, without Folium, a self-contained SVG map."""
    output = Path(output_html)
    try:
        import folium
    except ImportError:
        return _save_svg_map(gps, nodes, arcs, predicted_arc_ids, output, reference_arc_ids)
    center = [float(gps.latitude.mean()), float(gps.longitude.mean())]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    predicted = _route_coordinates(predicted_arc_ids, nodes, arcs)
    if predicted:
        folium.PolyLine(predicted, color="#1565c0", weight=5, opacity=0.9,
                        tooltip=f"Predicted route ({len(predicted_arc_ids)} arcs)").add_to(fmap)
    if reference_arc_ids:
        reference = _route_coordinates(reference_arc_ids, nodes, arcs)
        if reference:
            folium.PolyLine(reference, color="#e53935", weight=4, opacity=0.75,
                            dash_array="7, 7", tooltip=f"Reference route ({len(reference_arc_ids)} arcs)").add_to(fmap)
    points = gps[["latitude", "longitude"]].to_numpy().tolist()
    if points:
        folium.CircleMarker(points[0], radius=5, color="#2e7d32", fill=True, tooltip="Start GPS fix").add_to(fmap)
        folium.CircleMarker(points[-1], radius=5, color="#6a1b9a", fill=True, tooltip="End GPS fix").add_to(fmap)
        folium.PolyLine(points, color="#616161", weight=2, opacity=0.55,
                        tooltip="GPS observations").add_to(fmap)
    folium.LayerControl().add_to(fmap)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output))
    return output
