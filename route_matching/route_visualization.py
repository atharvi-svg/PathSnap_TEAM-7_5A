from __future__ import annotations

from pathlib import Path

try:
    import folium
    from folium import FeatureGroup, LayerControl, Map, PolyLine
except ImportError:  # Offline build hosts may not have the project dependencies installed.
    folium = None

from route_matching.road_graph import RoadGraph


def create_route_map(gps, graph: RoadGraph, reference_arc_ids, predicted_arc_ids,
                     output_path: str | Path, *, title: str = "PathSnap route comparison") -> Path:
    """Create a layered Folium map from an actual trip and decoded route."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if folium is None:
        return _create_static_fallback(gps, graph, reference_arc_ids, predicted_arc_ids, output_path, title)
    center = [float(gps.latitude.mean()), float(gps.longitude.mean())]
    fmap = Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    observed = FeatureGroup(name="GPS observations", show=True)
    coords = [[float(lat), float(lon)] for lon, lat in zip(gps.longitude, gps.latitude)]
    PolyLine(coords, color="#2463eb", weight=3, opacity=.75, tooltip="Observed GPS path").add_to(observed)
    for i in range(0, len(coords), max(1, len(coords) // 100)):
        folium.CircleMarker(coords[i], radius=2, color="#1746a2", fill=True, fill_opacity=.75,
                            tooltip=f"GPS observation {i}").add_to(observed)
    observed.add_to(fmap)

    road_layer = FeatureGroup(name="Nearby road network", show=False)
    lon_min, lon_max = float(gps.longitude.min()) - .003, float(gps.longitude.max()) + .003
    lat_min, lat_max = float(gps.latitude.min()) - .003, float(gps.latitude.max()) + .003
    selected = []
    for arc_id, (u, v) in enumerate(zip(graph.arc_from, graph.arc_to)):
        a, b = graph.nodes.iloc[int(u)], graph.nodes.iloc[int(v)]
        if (lon_min <= a.longitude <= lon_max and lat_min <= a.latitude <= lat_max) or (
                lon_min <= b.longitude <= lon_max and lat_min <= b.latitude <= lat_max):
            selected.append((int(u), int(v)))
    stride = max(1, len(selected) // 2500)
    for u, v in selected[::stride]:
        a, b = graph.nodes.iloc[u], graph.nodes.iloc[v]
        PolyLine([[float(a.latitude), float(a.longitude)], [float(b.latitude), float(b.longitude)]],
                 color="#9ca3af", weight=1, opacity=.45).add_to(road_layer)
    road_layer.add_to(fmap)

    ref_layer = FeatureGroup(name="Reference route (development only)", show=False)
    ref_coords = [[lat, lon] for lon, lat in graph.arc_coordinates(reference_arc_ids)]
    if len(ref_coords) > 1:
        PolyLine(ref_coords, color="#f59e0b", weight=5, opacity=.85, tooltip="Weak reference route").add_to(ref_layer)
    ref_layer.add_to(fmap)

    pred_layer = FeatureGroup(name="Predicted HMM route", show=True)
    pred_coords = [[lat, lon] for lon, lat in graph.arc_coordinates(predicted_arc_ids)]
    if len(pred_coords) > 1:
        PolyLine(pred_coords, color="#16a34a", weight=5, opacity=.9, tooltip="Viterbi route").add_to(pred_layer)
    pred_layer.add_to(fmap)
    LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(gps.latitude.min()), float(gps.longitude.min())],
                     [float(gps.latitude.max()), float(gps.longitude.max())]])
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;z-index:9999;top:12px;left:50px;background:white;padding:8px 12px;'
        f'border-radius:5px;font-weight:600;box-shadow:0 1px 5px #999">{title}</div>'))
    fmap.save(str(output_path))
    return output_path


def _create_static_fallback(gps, graph, reference_arc_ids, predicted_arc_ids, output_path, title):
    """Keep offline analysis possible; install folium to build the interactive map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from html import escape

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    lon_min, lon_max = float(gps.longitude.min()) - .003, float(gps.longitude.max()) + .003
    lat_min, lat_max = float(gps.latitude.min()) - .003, float(gps.latitude.max()) + .003
    for u, v in zip(graph.arc_from, graph.arc_to):
        a, b = graph.nodes.iloc[int(u)], graph.nodes.iloc[int(v)]
        if ((lon_min <= a.longitude <= lon_max and lat_min <= a.latitude <= lat_max) or
            (lon_min <= b.longitude <= lon_max and lat_min <= b.latitude <= lat_max)):
            ax.plot([a.longitude, b.longitude], [a.latitude, b.latitude], color="#cbd5e1", linewidth=.35, zorder=1)
    ax.plot(gps.longitude, gps.latitude, color="#60a5fa", linewidth=1.0, alpha=.35, label="GPS observations", zorder=2)
    for arc_ids, color, label, style in [(reference_arc_ids, "#f59e0b", "Reference route", "--"),
                                         (predicted_arc_ids, "#16a34a", "Predicted HMM route", "-")]:
        coords = graph.arc_coordinates(arc_ids)
        if coords:
            ax.plot([x[0] for x in coords], [x[1] for x in coords], color=color, linewidth=2.2,
                    linestyle=style,
                    alpha=.9, label=label, zorder=3)
    gps_sample = gps.iloc[::max(1, len(gps) // 180)]
    ax.scatter(gps_sample.longitude, gps_sample.latitude, s=7, color="#2563eb", alpha=.6,
               label="GPS fixes", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    # A simple Leaflet view is also written for browsers; the code switches to
    # the official Folium builder automatically when Folium is installed.
    gps_json = [[float(lon), float(lat)] for lon, lat in zip(gps.longitude, gps.latitude)]
    ref_json = [[float(lon), float(lat)] for lon, lat in graph.arc_coordinates(reference_arc_ids)]
    pred_json = [[float(lon), float(lat)] for lon, lat in graph.arc_coordinates(predicted_arc_ids)]
    import json
    html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{escape(title)}</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <style>html,body,#map{{height:100%;margin:0}} .legend{{background:#fff;padding:8px;font:14px sans-serif}}</style></head><body><div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
    const gps={json.dumps(gps_json)}, ref={json.dumps(ref_json)}, pred={json.dumps(pred_json)};
    const map=L.map('map'); L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(map);
    const layers={{'GPS observations':L.polyline(gps.map(p=>[p[1],p[0]]),{{color:'#2563eb',weight:3}}),
    'Reference route':L.polyline(ref.map(p=>[p[1],p[0]]),{{color:'#f59e0b',weight:5}}),
    'Predicted HMM route':L.polyline(pred.map(p=>[p[1],p[0]]),{{color:'#16a34a',weight:5}})}};
    layers['GPS observations'].addTo(map); layers['Predicted HMM route'].addTo(map); L.control.layers(null,layers).addTo(map);
    map.fitBounds(layers['GPS observations'].getBounds());</script></body></html>'''
    output_path.write_text(html, encoding="utf-8")
    return output_path
