//! gmsh MSH 2.2 reader: nodes + TET10 (type 11) / TET4 (type 4).
use std::collections::HashMap;
use std::path::Path;

pub const TET4_TYPE: u32 = 4;
pub const TET10_TYPE: u32 = 11;

#[derive(Debug, Clone)]
pub struct TetMesh {
    pub nodes: Vec<[f64; 3]>,      // (n,3) meters
    pub elements: Vec<[usize; 10]>, // T10 conn; order 1 uses first 4
    pub order: u8,
}

pub fn read_msh22(path: &Path) -> Result<TetMesh, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut nodes_raw: Vec<(u64, [f64; 3])> = Vec::new();
    let mut tets4: Vec<Vec<u64>> = Vec::new();
    let mut tets10: Vec<Vec<u64>> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let t = line.trim();
        if t == "$Nodes" {
            let n: usize = lines
                .next()
                .and_then(|s| s.trim().parse().ok())
                .ok_or("bad $Nodes count")?;
            for _ in 0..n {
                let l = lines.next().ok_or("node line missing")?;
                let mut it = l.split_whitespace();
                let id: u64 = it.next().and_then(|s| s.parse().ok()).ok_or("bad node id")?;
                let x: f64 = it.next().and_then(|s| s.parse().ok()).ok_or("bad node x")?;
                let y: f64 = it.next().and_then(|s| s.parse().ok()).ok_or("bad node y")?;
                let z: f64 = it.next().and_then(|s| s.parse().ok()).ok_or("bad node z")?;
                nodes_raw.push((id, [x, y, z]));
            }
        } else if t == "$Elements" {
            let n: usize = lines
                .next()
                .and_then(|s| s.trim().parse().ok())
                .ok_or("bad $Elements count")?;
            for _ in 0..n {
                let l = lines.next().ok_or("element line missing")?;
                let parts: Vec<&str> = l.split_whitespace().collect();
                let el_type: u32 = parts[1].parse().map_err(|_| "bad el type")?;
                let ntags: usize = parts[2].parse().map_err(|_| "bad ntags")?;
                let conn: Vec<u64> = parts[3 + ntags..]
                    .iter()
                    .map(|s| s.parse::<u64>().map_err(|_| "bad conn"))
                    .collect::<Result<_, _>>()?;
                match el_type {
                    TET10_TYPE => tets10.push(conn),
                    TET4_TYPE => tets4.push(conn),
                    _ => {}
                }
            }
        }
    }
    let mut id_map: HashMap<u64, usize> = HashMap::with_capacity(nodes_raw.len());
    let mut nodes: Vec<[f64; 3]> = Vec::with_capacity(nodes_raw.len());
    for (i, (id, xyz)) in nodes_raw.iter().enumerate() {
        id_map.insert(*id, i);
        nodes.push(*xyz);
    }
    let (order, raw) = if !tets10.is_empty() {
        (2u8, tets10)
    } else if !tets4.is_empty() {
        (1u8, tets4)
    } else {
        return Err("no tetrahedra in mesh".into());
    };
    let mut elements = Vec::with_capacity(raw.len());
    for conn in &raw {
        let mut e = [0usize; 10];
        for (k, p) in conn.iter().enumerate() {
            e[k] = *id_map.get(p).ok_or("conn refers to unknown node")?;
        }
        if order == 2 {
            // gmsh tet10 mid-edge order (0-1,1-2,2-0,0-3,2-3,1-3) -> solver/VTK
            // convention (0-1,1-2,2-0,0-3,1-3,2-3): swap last two.
            e.swap(8, 9);
        }
        elements.push(e);
    }
    Ok(TetMesh { nodes, elements, order })
}
