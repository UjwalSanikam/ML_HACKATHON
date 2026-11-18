"""Folium map helpers (shared by EDA, evaluation and the report)."""
from __future__ import annotations

import folium
import numpy as np
from folium.plugins import Fullscreen, MeasureControl


def _lines(net, arc_ids):
    return [[net.nodes_lonlat[u, ::-1].tolist(), net.nodes_lonlat[v, ::-1].tolist()]
            for u, v in net.arcs[np.asarray(arc_ids, int)]]


def trip_map(net, track_lonlat, reference=None, predicted=None, classifier_only=None,
             network_arcs=None, title="", obs_step=1, zoom_start=15):
    lat, lon = track_lonlat[:, 1], track_lonlat[:, 0]
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=zoom_start,
                   tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    if network_arcs is not None:
        fg = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
        folium.PolyLine(_lines(net, network_arcs), color="#9e9e9e", weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
    if reference is not None and len(reference):
        fg = folium.FeatureGroup(name="Reference route (ground truth)")
        folium.PolyLine(_lines(net, reference), color="#2e7d32", weight=7, opacity=0.55,
                        tooltip="Reference route").add_to(fg)
        fg.add_to(m)
    if classifier_only is not None and len(classifier_only):
        fg = folium.FeatureGroup(name="Classifier-only (top-1 per point)", show=False)
        folium.PolyLine(_lines(net, classifier_only), color="#ef6c00", weight=3, opacity=0.8,
                        tooltip="Classifier only").add_to(fg)
        fg.add_to(m)
    if predicted is not None and len(predicted):
        fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
        folium.PolyLine(_lines(net, predicted), color="#d81b60", weight=3.5, opacity=0.95,
                        tooltip="Predicted route").add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS observations")
    for p in track_lonlat[::obs_step]:
        folium.CircleMarker([p[1], p[0]], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            weight=0).add_to(fg)
    fg.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index:9999; background: white;
         padding: 10px 14px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.3); font: 13px sans-serif;">
      <b>{title}</b><br>
      <span style="color:#1565c0">&#9679;</span> Raw GPS<br>
      <span style="color:#2e7d32">&#9644;</span> Reference route<br>
      <span style="color:#ef6c00">&#9644;</span> Classifier only<br>
      <span style="color:#d81b60">&#9644;</span> Classifier + HMM<br>
      <span style="color:#9e9e9e">&#9644;</span> Road network
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl().add_to(m)
    return m
