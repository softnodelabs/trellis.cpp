// Narrow-band UDF dual-contouring remesh, ported from the reference
// (CuMesh remeshing.py::remesh_narrow_band_dc + simple_dual_contour.cu; see
// docs/spec/27-reference-postprocess.md §4). Rebuilds a noisy voxel-derived
// mesh as the closed, consistently-oriented offset surface at distance
// eps = band·scale/res around it — one watertight manifold that downstream
// simplification and UV unwrapping can digest.
#pragma once
#include "dual_grid.h"

namespace trellis {

class TriBvh;

// `bvh` must be built over (verts, faces). Returns an empty mesh on failure
// (caller keeps the un-remeshed path).
// `project_back` lerps each dual vertex toward its closest point on the input
// surface (o_voxel postprocess `remesh_project`; 0 disables).
//
// Default 0 to match the HF space — the config that produced the reference GLBs
// users compare against — which passes remesh_project=0 (trellis2-space/app.py).
// o_voxel's own to_glb default is 0.9 and TRELLIS.2's app.py inherits it, so the
// knob is kept: measured on the issue #1 goblin, 0.9 improves the surface
// (dihedral 8.08->7.41deg, >60deg spikes 1.12%->0.64%) but quadruples our UV
// merge clusters (2519->10248), so it is NOT a free win in this pipeline.
//
// `fill_inside` keeps only the outer surface. The band around a closed solid has two boundaries,
// the offset surface outside it and a second one eps inside it, so without this every solid comes
// out as a hollow shell with a hidden inner skin holding about half the faces. With it, the
// empty space around the band surface is split, on a coarse grid, into the outside and the
// enclosed spaces (openings narrower than about two coarse cells count as closed), and each
// enclosed space with room in it is voted inside or out as a whole by ray parity against the
// input, which does not depend on the input's winding. The faces that look into a space voted
// inside are the inner skin and are removed, the rims this leaves where an opening of the input
// joined the two skins are capped, and hollows left sealed inside the solid are dropped. Points
// beside an open sheet see it on one side only and vote outside, so an open sheet keeps its
// two-sided shell as before.
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, int band = 1,
                           float project_back = 0.0f, bool fill_inside = false);

}  // namespace trellis
