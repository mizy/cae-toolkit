#!/usr/bin/env python3
"""Local model/condition preparation service; equation solving stays on the device."""
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse
import base64
import json
from pathlib import Path
import tempfile

import numpy as np
from export_web_case import export_case
from stl_volume import mesh_stl
import trimesh

WEB = Path(__file__).resolve().parents[1] / "web"


def prepare_case(model: dict, conditions: dict) -> dict:
    """Own automatic meshing choices for currently supported solid models."""
    geometry, units = model["geometry"], model["units"]
    mesh = None
    if units not in ("m", "mm"):
        raise ValueError("模型单位必须为 m 或 mm")
    if geometry == "stl":
        content = base64.b64decode(model["stl_base64"], validate=True)
        with tempfile.TemporaryDirectory(prefix="structural-upload-") as folder:
            source = Path(folder) / "input.stl"
            source.write_bytes(content)
            surface = trimesh.load_mesh(source, force='mesh', process=True)
            dims = np.ptp(surface.vertices,axis=0)
            if np.any(dims <= 0) or not np.isfinite(dims).all():
                raise ValueError("STL has invalid dimensions")
            mesh_size = float(dims.min()/2)
            mesh = mesh_stl(source, units, mesh_size)
    elif geometry.startswith("box:"):
        dims = np.asarray([float(v) for v in geometry[4:].split(",")])
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            raise ValueError("长宽高必须为三个正数")
        mesh_size = float(dims.min() / 2)
        if np.prod(dims / mesh_size) > 20000:
            raise ValueError("模型长宽厚比例超出当前自动实体网格范围；请分区或使用经过验证的体网格")
    elif geometry == "hinge" and units == "mm":
        mesh_size = 3.0
    else:
        raise ValueError("当前支持 box:L,W,H 或 mm 单位下的简化 hinge 实体")
    with tempfile.TemporaryDirectory(prefix="structural-case-") as folder:
        path = Path(folder) / "case.json"
        export_case(geometry, units, mesh_size, 2, conditions["material"],
                    conditions["fixes"], conditions.get("loads", []), path, mesh=mesh,
                    gravity=conditions.get("gravity"), density=conditions.get("density"), cylinder=conditions.get("cylinder"))
        case = json.loads(path.read_text())
    case["model"] = {k:v for k,v in model.items() if k != "stl_base64"}
    case["conditions"] = conditions
    case["metadata"]["representation"] = "solid T10"
    case["metadata"]["mesh_selection"] = "automatic, initial mesh; no mesh-convergence claim"
    return case


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/prepare":
            self.send_error(404)
            return
        # Only the same local page may invoke expensive meshing; no remote files or commands.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16*1024*1024:
                raise ValueError("模型和条件请求过大或为空")
            request = json.loads(self.rfile.read(length))
            response = prepare_case(request["model"], request["conditions"])
            body = json.dumps(response, allow_nan=False).encode()
            self.send_response(200)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            body = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18768)
    args = parser.parse_args()
    HTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(WEB))).serve_forever()
