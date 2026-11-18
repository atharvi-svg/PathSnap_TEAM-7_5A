from __future__ import annotations

import json
from html import escape


def route_map_html(gps_coordinates, route_coordinates, *, trip_id: str) -> str:
    if not gps_coordinates:
        raise ValueError("The API returned no GPS coordinates")
    try:
        import folium
    except ImportError:
        all_points = gps_coordinates + route_coordinates
        lons = [p[0] for p in all_points]
        lats = [p[1] for p in all_points]
        lo0, hi0 = min(lons), max(lons)
        la0, hi1 = min(lats), max(lats)
        sx = max(hi0 - lo0, 1e-9)
        sy = max(hi1 - la0, 1e-9)
        def svg_points(points):
            return " ".join(f"{24 + 852 * (x - lo0) / sx:.1f},{476 - 436 * (y - la0) / sy:.1f}" for x, y in points)
        markers = "".join(f'<circle cx="{24 + 852*(x-lo0)/sx:.1f}" cy="{476-436*(y-la0)/sy:.1f}" r="2" fill="#1d4ed8" opacity=".6"/>'
                          for x, y in gps_coordinates[::max(1, len(gps_coordinates)//120)])
        return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>PathSnap trip {escape(trip_id)}</title><style>body{{font:15px Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}header{{padding:12px 18px;background:#fff;border-bottom:1px solid #cbd5e1;font-weight:700}}svg{{width:100%;height:calc(100vh - 52px);background:#fff}}.gps{{fill:none;stroke:#60a5fa;stroke-width:2;opacity:.7}}.route{{fill:none;stroke:#16a34a;stroke-width:4}}</style></head><body>
        <header>Trip {escape(trip_id)} — observed GPS and predicted HMM route</header><svg viewBox="0 0 900 500" role="img" aria-label="GPS trajectory and predicted route">
        <polyline class="gps" points="{svg_points(gps_coordinates)}"/>{markers}<polyline class="route" points="{svg_points(route_coordinates)}"/>
        <g font-family="Arial" font-size="13"><rect x="35" y="20" width="235" height="54" rx="6" fill="white" stroke="#cbd5e1"/><line x1="50" y1="40" x2="80" y2="40" stroke="#60a5fa" stroke-width="3"/><text x="88" y="44">GPS observations</text><line x1="50" y1="58" x2="80" y2="58" stroke="#16a34a" stroke-width="4"/><text x="88" y="62">Predicted HMM route</text></g></svg></body></html>'''
    center = [sum(p[1] for p in gps_coordinates) / len(gps_coordinates),
              sum(p[0] for p in gps_coordinates) / len(gps_coordinates)]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True)
    gps_layer = folium.FeatureGroup(name="GPS observations")
    gps_latlon = [[p[1], p[0]] for p in gps_coordinates]
    folium.PolyLine(gps_latlon, color="#2563eb", weight=3, opacity=.75,
                    tooltip="Observed GPS trajectory").add_to(gps_layer)
    for i in range(0, len(gps_latlon), max(1, len(gps_latlon) // 100)):
        folium.CircleMarker(gps_latlon[i], radius=2, color="#1d4ed8", fill=True,
                            fill_opacity=.75, tooltip=f"Observation {i}").add_to(gps_layer)
    gps_layer.add_to(fmap)
    if gps_latlon:
        folium.CircleMarker(gps_latlon[0], radius=7, color="#ffffff", weight=2,
                            fill=True, fill_color="#178b69", fill_opacity=1,
                            tooltip="Trip start").add_to(fmap)
        folium.CircleMarker(gps_latlon[-1], radius=7, color="#ffffff", weight=2,
                            fill=True, fill_color="#d66b43", fill_opacity=1,
                            tooltip="Trip end").add_to(fmap)

    predicted = folium.FeatureGroup(name="Predicted HMM route")
    if route_coordinates:
        folium.PolyLine([[p[1], p[0]] for p in route_coordinates], color="#16a34a", weight=5,
                        opacity=.9, tooltip="Viterbi route").add_to(predicted)
    predicted.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[1] for p in gps_coordinates), min(p[0] for p in gps_coordinates)],
                     [max(p[1] for p in gps_coordinates), max(p[0] for p in gps_coordinates)]])
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;z-index:9999;top:10px;left:50px;background:white;padding:8px 12px;'
        f'font-weight:600;border-radius:5px">Trip {trip_id}: GPS and predicted route</div>'))
    return fmap.get_root().render()
