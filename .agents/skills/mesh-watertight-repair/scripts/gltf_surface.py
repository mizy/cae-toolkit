"""Decode glTF triangles and native normals with accessor and node transforms."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pyvista as pv
import trimesh


def read_gltf_surface(path: Path) -> pv.PolyData:
    blob = path.read_bytes()
    binary = None
    if path.suffix.lower() == ".glb":
        magic, version, length = struct.unpack_from("<III", blob)
        if magic != 0x46546C67 or version != 2 or length != len(blob):
            raise ValueError("expected a complete glTF 2 GLB")
        offset = 12
        document = None
        while offset < len(blob):
            size, kind = struct.unpack_from("<II", blob, offset)
            chunk = blob[offset + 8 : offset + 8 + size]
            if len(chunk) != size:
                raise ValueError("truncated GLB chunk")
            if kind == 0x4E4F534A:
                document = json.loads(chunk)
            elif kind == 0x004E4942:
                binary = chunk
            offset += 8 + size
        if document is None:
            raise ValueError("GLB JSON chunk is missing")
    else:
        document = json.loads(blob)
    buffers = []
    for spec in document.get("buffers", []):
        uri = spec.get("uri")
        if uri is None:
            data = binary
        elif uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        elif "://" in uri:
            raise ValueError(
                "remote glTF buffers must be downloaded explicitly before meshing"
            )
        else:
            data = (path.parent / unquote(uri)).read_bytes()
        if data is None or len(data) < spec["byteLength"]:
            raise ValueError("glTF buffer is missing or truncated")
        buffers.append(data)
    access = lambda index: gltf_accessor(document, buffers, index)
    positions, triangles, node_ids, normal_sets = [], [], [], []
    count = 0

    def walk(index: int, parent: np.ndarray, ancestry: set[int]) -> None:
        nonlocal count
        if index in ancestry:
            raise ValueError("cyclic glTF node hierarchy")
        node = document["nodes"][index]
        if "skin" in node or any(node.get("weights", [])):
            raise ValueError(
                "bake glTF skinning and morph weights before surface meshing"
            )
        if "matrix" in node:
            local = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            local = trimesh.transformations.quaternion_matrix([w, x, y, z])
            local[:3, :3] = local[:3, :3] @ np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        transform = parent @ local
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            if any(mesh.get("weights", [])):
                raise ValueError("bake glTF morph weights before surface meshing")
            for primitive in mesh["primitives"]:
                if primitive.get("extensions", {}).get("KHR_draco_mesh_compression"):
                    raise ValueError("decode Draco geometry before meshing")
                mode = primitive.get("mode", 4)
                if mode not in (4, 5, 6):
                    continue
                points = trimesh.transform_points(
                    access(primitive["attributes"]["POSITION"]), transform
                )
                indices = (
                    access(primitive["indices"]).ravel().astype(np.int64)
                    if "indices" in primitive
                    else np.arange(len(points))
                )
                if mode == 4:
                    faces = indices.reshape(-1, 3)
                elif mode == 5:
                    faces = np.column_stack([indices[:-2], indices[1:-1], indices[2:]])
                    faces[1::2] = faces[1::2, ::-1]
                else:
                    faces = np.column_stack(
                        [
                            np.full(len(indices) - 2, indices[0]),
                            indices[1:-1],
                            indices[2:],
                        ]
                    )
                if (
                    np.any(faces < 0)
                    or np.any(faces >= len(points))
                    or not np.isfinite(points).all()
                ):
                    raise ValueError("invalid glTF positions or indices")
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                normal_index = primitive["attributes"].get("NORMAL")
                normals = None
                if normal_index is not None:
                    normals = np.asarray(access(normal_index), dtype=float)
                    if normals.shape != points.shape or not np.isfinite(normals).all():
                        raise ValueError("invalid glTF vertex normals")
                    normals = normals @ np.linalg.inv(transform[:3, :3])
                    lengths = np.linalg.norm(normals, axis=1)
                    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
                        raise ValueError("glTF normal transform must be finite and nonzero")
                    normals /= lengths[:, None]
                positions.append(points)
                triangles.append(faces + count)
                node_ids.append(np.full(len(faces), index, dtype=np.int32))
                normal_sets.append(normals)
                count += len(points)
        for child in node.get("children", []):
            walk(child, transform, ancestry | {index})

    nodes = document.get("scenes", [{}])[document.get("scene", 0)].get("nodes")
    if nodes is None:
        children = {
            child
            for node in document.get("nodes", [])
            for child in node.get("children", [])
        }
        nodes = [i for i in range(len(document.get("nodes", []))) if i not in children]
    for root in nodes:
        walk(root, np.eye(4), set())
    if not triangles:
        raise ValueError("glTF contains no supported triangle surfaces")
    faces = np.vstack(triangles)
    result = pv.PolyData(
        np.vstack(positions), np.column_stack([np.full(len(faces), 3), faces]).ravel()
    )
    result.cell_data["gltf_node_index"] = np.concatenate(node_ids)
    if any(normals is not None for normals in normal_sets):
        fallback = None
        if any(normals is None for normals in normal_sets):
            fallback = result.compute_normals(
                cell_normals=False, split_vertices=False,
            ).point_data["Normals"]
        offset, chunks = 0, []
        for points, normals in zip(positions, normal_sets):
            chunks.append(normals if normals is not None else fallback[offset:offset + len(points)])
            offset += len(points)
        result.point_data["Normals"] = np.vstack(chunks).astype(np.float32)
        result.GetPointData().SetActiveNormals("Normals")
    return result


def gltf_accessor(document: dict, buffers: list[bytes], index: int) -> np.ndarray:
    spec = document["accessors"][index]
    dtype = np.dtype(
        {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[
            spec["componentType"]
        ]
    )
    columns = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[spec["type"]]

    def view_values(
        view_index: int, count: int, size: int, kind: np.dtype, offset: int = 0
    ):
        view = document["bufferViews"][view_index]
        if "EXT_meshopt_compression" in view.get("extensions", {}):
            raise ValueError("decode meshopt compression before meshing")
        stride = view.get("byteStride", size * kind.itemsize)
        end = offset + max(0, count - 1) * stride + size * kind.itemsize
        if count and end > view["byteLength"]:
            raise ValueError("glTF accessor exceeds its buffer view")
        return np.ndarray(
            (count, size),
            dtype=kind,
            buffer=buffers[view["buffer"]],
            offset=view.get("byteOffset", 0) + offset,
            strides=(stride, kind.itemsize),
        ).copy()

    result = (
        view_values(
            spec["bufferView"], spec["count"], columns, dtype, spec.get("byteOffset", 0)
        )
        if "bufferView" in spec
        else np.zeros((spec["count"], columns), dtype=dtype)
    )
    if sparse := spec.get("sparse"):
        indices = sparse["indices"]
        index_type = np.dtype(
            {5121: "u1", 5123: "<u2", 5125: "<u4"}[indices["componentType"]]
        )
        ids = view_values(
            indices["bufferView"],
            sparse["count"],
            1,
            index_type,
            indices.get("byteOffset", 0),
        ).ravel()
        values = sparse["values"]
        result[ids] = view_values(
            values["bufferView"],
            sparse["count"],
            columns,
            dtype,
            values.get("byteOffset", 0),
        )
    if spec.get("normalized") and dtype.kind in "iu":
        result = result.astype(float) / np.iinfo(dtype).max
        if dtype.kind == "i":
            result = np.maximum(result, -1.0)
    return result
