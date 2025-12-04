from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

DATA_FILE = Path("embeddings_index.npz")
META_FILE = Path("embeddings_meta.json")
DEFAULT_OUTPUT = Path("embeddings_view.html")


def _load_embeddings(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    npz = np.load(npz_path)
    if not npz.files:
        raise ValueError(f"No arrays stored in {npz_path}")
    return npz[npz.files[0]]


def _load_labels(meta_path: Path, expected: int) -> List[str]:
    if not meta_path.exists():
        raise FileNotFoundError(f"JSON metadata not found: {meta_path}")
    items = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(items) != expected:
        raise ValueError(
            f"Metadata count {len(items)} does not match embeddings {expected}"
        )
    return [str(entry.get("filename", f"item {idx}")) for idx, entry in enumerate(items)]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("Expected a 2D array of embeddings")
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # Fast SVD-based PCA to avoid extra dependencies.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def _build_html(points: Sequence[tuple[float, float]], labels: Sequence[str]) -> str:
    payload = [
        {"x": float(x), "y": float(y), "label": label}
        for (x, y), label in zip(points, labels)
    ]
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Embeddings view</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const points = {data_json};
    const trace = {{
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      text: points.map(p => p.label),
      hovertemplate: "%{{text}}<extra></extra>",
      mode: "markers",
      type: "scattergl",
      marker: {{
        size: 8,
        color: points.map(p => p.y),
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.85
      }}
    }};
    const layout = {{
      title: "Embeddings view",
      xaxis: {{ title: "PC 1" }},
      yaxis: {{ title: "PC 2" }},
      hovermode: "closest",
      margin: {{ l: 60, r: 20, t: 60, b: 60 }},
      paper_bgcolor: "#f7f7f7",
      plot_bgcolor: "#f7f7f7"
    }};
    Plotly.newPlot("plot", [trace], layout, {{responsive: true}});
  </script>
</body>
</html>
"""


def main(output_path: Path) -> None:
    embeddings = _load_embeddings(DATA_FILE)
    labels = _load_labels(META_FILE, embeddings.shape[0])
    coords = _pca_2d(embeddings)
    # Zet ndarray om naar lijst van tuples om typehint te matchen
    coords_tuples = [tuple(row) for row in coords]
    html = _build_html(coords_tuples, labels)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote view to {output_path.resolve()}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    main(target)

