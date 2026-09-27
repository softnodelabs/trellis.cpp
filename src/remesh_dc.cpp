#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> ts;
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e);
    }
    for (auto& t : ts) t.join();
}

// Rays per inside vote, and how many of them must cross the input an odd number of times. The
// input is a decoded surface with doubled sheets and small holes, so a single ray's parity is not
// to be trusted; a point beside an open sheet sees it over at most half the directions, so a large
// majority keeps sheets out.
constexpr int VOTE_RAYS = 8;
constexpr int VOTE_INSIDE = 6;

// The cube's diagonals turned off the grid axes, so a ray rarely runs along a face or through an
// edge of the voxel-aligned input.
const float VOTE_DIRS[VOTE_RAYS][3] = {
    { 0.5774f,  0.5919f,  0.5625f},
    {-0.6293f,  0.4183f,  0.6550f},
    { 0.3419f, -0.7121f,  0.6132f},
    { 0.6427f,  0.4861f, -0.5922f},
    {-0.4481f, -0.5397f, -0.7127f},
    {-0.5534f,  0.6012f, -0.5764f},
    { 0.6161f, -0.5102f, -0.6001f},
    {-0.5902f, -0.6268f,  0.5086f},
};

// Whether `p` lies inside the input, by ray parity over VOTE_RAYS directions. A ray that meets
// nothing at all settles it: nothing enclosed can see out, whatever the parity of the others. A
// point in a pocket of the outer surface (a chest inside a collar) is often ringed by doubled
// sheets that fool the parity, but always sees out somewhere.
bool vote_inside(const TriBvh& bvh, const float p[3]) {
    int odd = 0;
    for (int r = 0; r < VOTE_RAYS; ++r) {
        const int crossings = bvh.count_crossings(p, VOTE_DIRS[r]);
        if (crossings == 0) return false;
        odd += crossings & 1;
        if (odd + (VOTE_RAYS - 1 - r) < VOTE_INSIDE) return false;
    }
    return odd >= VOTE_INSIDE;
}

// Removes the pieces that face inwards all over: the walls of a hollow sealed inside the solid
// that the vote kept (the inside of an open tube, whose rays escape through its ends). The surface
// is wound outwards from the band, so such a piece encloses a negative volume, and nothing outside
// the solid can see it. Returns the faces removed.
int64_t drop_hollows(Mesh& mesh) {
    const int64_t F = (int64_t)mesh.faces.size() / 3;
    const int32_t V = (int32_t)(mesh.verts.size() / 3);
    std::vector<int32_t> parent((size_t)V);
    for (int32_t v = 0; v < V; ++v) parent[v] = v;
    auto find = [&parent](int32_t v) {
        while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; }
        return v;
    };
    for (int64_t f = 0; f < F; ++f)
        for (int k = 1; k < 3; ++k) {
            const int32_t a = find(mesh.faces[3*f]), b = find(mesh.faces[3*f+k]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    std::unordered_map<int32_t, double> volume;
    for (int64_t f = 0; f < F; ++f) {
        const float* a = &mesh.verts[3 * (size_t)mesh.faces[3*f]];
        const float* b = &mesh.verts[3 * (size_t)mesh.faces[3*f+1]];
        const float* c = &mesh.verts[3 * (size_t)mesh.faces[3*f+2]];
        // a . (b x c) / 6, the signed volume of the tetrahedron with the origin.
        volume[find(mesh.faces[3*f])] += ((double)a[0] * ((double)b[1]*c[2] - (double)b[2]*c[1])
                                        + (double)a[1] * ((double)b[2]*c[0] - (double)b[0]*c[2])
                                        + (double)a[2] * ((double)b[0]*c[1] - (double)b[1]*c[0])) / 6.0;
    }
    std::vector<int32_t> kept;
    kept.reserve(mesh.faces.size());
    for (int64_t f = 0; f < F; ++f)
        if (volume[find(mesh.faces[3*f])] >= 0.0) kept.insert(kept.end(), &mesh.faces[3*f], &mesh.faces[3*f] + 3);
    const int64_t dropped = F - (int64_t)kept.size() / 3;
    mesh.faces.swap(kept);
    return dropped;
}

// A patch of faces smaller than this is taken to be vote noise and joins what surrounds it.
constexpr size_t NOISE_FACES = 64;

// How each half-edge's edge is used: half-edge 3f+k runs from corner k of face f to corner k+1.
// `count` is how many faces use its undirected edge (at most 255), and `twin` the other half-edge
// when exactly two do. Built by sorting, in buckets by the edge's lower vertex, rather than with
// a hash map, which on the band surface's 20M half-edges took several times as long.
struct EdgeUse {
    std::vector<uint8_t> count;
    std::vector<int64_t> twin;

    static EdgeUse of(const std::vector<int32_t>& faces) {
        EdgeUse u;
        const int64_t H = (int64_t)faces.size();
        u.count.assign((size_t)H, 0);
        u.twin.assign((size_t)H, -1);
        auto key = [&faces](int64_t h) {
            const int64_t f = h / 3;
            int32_t a = faces[(size_t)h], b = faces[(size_t)(3 * f + (h - 3 * f + 1) % 3)];
            if (a > b) std::swap(a, b);
            return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        };
        const int buckets = 256;
        auto bucket_of = [](uint64_t k) { return (int)((k >> 32) % 256); };
        std::vector<int64_t> start(buckets + 1, 0);
        for (int64_t h = 0; h < H; ++h) ++start[bucket_of(key(h)) + 1];
        for (int b = 0; b < buckets; ++b) start[b + 1] += start[b];
        std::vector<std::pair<uint64_t, int64_t>> keys((size_t)H);
        {
            std::vector<int64_t> at(start.begin(), start.end() - 1);
            for (int64_t h = 0; h < H; ++h) {
                const uint64_t k = key(h);
                keys[(size_t)at[bucket_of(k)]++] = {k, h};
            }
        }
        parallel_for(buckets, [&](int64_t b0, int64_t b1) {
            for (int64_t b = b0; b < b1; ++b) {
                std::sort(keys.begin() + start[b], keys.begin() + start[b + 1]);
                for (int64_t i = start[b]; i < start[b + 1];) {
                    int64_t j = i;
                    while (j < start[b + 1] && keys[(size_t)j].first == keys[(size_t)i].first) ++j;
                    const uint8_t n = (uint8_t)std::min<int64_t>(255, j - i);
                    for (int64_t k = i; k < j; ++k) u.count[(size_t)keys[(size_t)k].second] = n;
                    if (j - i == 2) {
                        u.twin[(size_t)keys[(size_t)i].second] = keys[(size_t)i + 1].second;
                        u.twin[(size_t)keys[(size_t)i + 1].second] = keys[(size_t)i].second;
                    }
                    i = j;
                }
            }
        });
        return u;
    }
};

// Closes every open loop of `mesh`; returns how many. The loops are traced over the edges used
// once, each walked opposite to its face, so the caps are wound like the faces around them. Unlike `fill_holes` a loop may pass a vertex where two rims touch (the
// dual contour has a few non-manifold spots): the walk takes any unused way on.
int cap_loops(Mesh& mesh) {
    const int64_t F = (int64_t)mesh.faces.size() / 3;
    const int64_t V = (int64_t)mesh.verts.size() / 3;
    const EdgeUse use = EdgeUse::of(mesh.faces);
    // The rim edges, each walked opposite to its face (from v to u for a face edge u -> v), as
    // runs per start vertex; `pos` is the next unused one of each run.
    std::vector<int64_t> first((size_t)V + 1, 0);
    for (int64_t h = 0; h < 3 * F; ++h)
        if (use.count[(size_t)h] == 1) ++first[(size_t)mesh.faces[(size_t)(h - h % 3 + (h % 3 + 1) % 3)] + 1];
    for (int64_t v = 0; v < V; ++v) first[(size_t)v + 1] += first[(size_t)v];
    std::vector<int32_t> to((size_t)first[(size_t)V]);
    std::vector<int64_t> pos(first.begin(), first.end() - 1);
    for (int64_t h = 0; h < 3 * F; ++h)
        if (use.count[(size_t)h] == 1) {
            const int32_t u = mesh.faces[(size_t)h], v = mesh.faces[(size_t)(h - h % 3 + (h % 3 + 1) % 3)];
            to[(size_t)pos[(size_t)v]++] = u;
        }
    std::copy(first.begin(), first.end() - 1, pos.begin());
    auto take = [&](int32_t v) -> int32_t {
        return pos[(size_t)v] < first[(size_t)v + 1] ? to[(size_t)pos[(size_t)v]++] : -1;
    };
    int capped = 0;
    // A loop is closed by cutting off, again and again, the corner with the smallest angle. A
    // rim is often a long, thin, crooked slot (where a gap between a garment and the body opens
    // at the collar); a fan from its centroid spans the slot with long crossing slivers, while
    // cutting the sharpest corners first zips it shut from its ends. Each cut triangle follows
    // the loop's direction, so it is wound like the faces round the rim.
    auto cap = [&mesh, &capped](const int32_t* loop, size_t n) {
        if (n < 3) return;
        std::vector<int32_t> ring(loop, loop + n);
        auto angle = [&mesh](int32_t a, int32_t b, int32_t c) {
            const float* pa = &mesh.verts[3 * (size_t)a];
            const float* pb = &mesh.verts[3 * (size_t)b];
            const float* pc = &mesh.verts[3 * (size_t)c];
            double u[3], v[3], uu = 0, vv = 0, uv = 0;
            for (int k = 0; k < 3; ++k) {
                u[k] = (double)pa[k] - pb[k]; v[k] = (double)pc[k] - pb[k];
                uu += u[k] * u[k]; vv += v[k] * v[k]; uv += u[k] * v[k];
            }
            if (uu <= 0 || vv <= 0) return 0.0;
            return std::acos(std::max(-1.0, std::min(1.0, uv / std::sqrt(uu * vv))));
        };
        while (ring.size() > 3) {
            const size_t m = ring.size();
            size_t best = 0;
            double best_angle = 1e30;
            for (size_t i = 0; i < m; ++i) {
                const double a = angle(ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]);
                if (a < best_angle) { best_angle = a; best = i; }
            }
            mesh.faces.push_back(ring[(best + m - 1) % m]);
            mesh.faces.push_back(ring[best]);
            mesh.faces.push_back(ring[(best + 1) % m]);
            ring.erase(ring.begin() + (std::ptrdiff_t)best);
        }
        mesh.faces.insert(mesh.faces.end(), ring.begin(), ring.end());
        ++capped;
    };
    // Each walk closes a loop as soon as it comes back to a vertex already on it, and goes on from
    // there; a walk that stops short (a vertex with more rim edges in than out) leaves its open
    // part as it is.
    std::vector<int32_t> path;
    std::unordered_map<int32_t, size_t> at;
    for (int32_t s = 0; s < (int32_t)V; ++s) {
        while (pos[(size_t)s] < first[(size_t)s + 1]) {
            path.assign(1, s);
            at.clear();
            at[s] = 0;
            int32_t cur = s;
            while (true) {
                cur = take(cur);
                if (cur < 0) break;
                auto seen = at.find(cur);
                if (seen == at.end()) {
                    at[cur] = path.size();
                    path.push_back(cur);
                    continue;
                }
                const size_t j = seen->second;
                cap(&path[j], path.size() - j);
                for (size_t i = j + 1; i < path.size(); ++i) at.erase(path[i]);
                path.resize(j + 1);
            }
        }
    }
    return capped;
}

// The coarse grid is this many band cells to a side of its own cells.
constexpr int SPACE_COARSEN = 4;
// Openings narrower than twice this many coarse cells are closed while the outside is flooded,
// so a hollow reached only through an eye socket or the open end of a limb counts as enclosed.
constexpr int SPACE_CLOSE = 2;
// Samples per enclosed space, and the share of them that must vote inside.
constexpr int SPACE_SAMPLES = 24;
constexpr float SPACE_INSIDE = 0.75f;
// Cells clear of every wall an enclosed space needs before it can count as a hollow.
constexpr int SPACE_DEEP = 8;

// The empty space around and within the band surface, on a coarse grid, told apart into the
// outside and the enclosed spaces, each of those voted inside or out as a whole. One face's vote
// is too noisy on a decoded input with doubled sheets: a chest in a collar's pocket can vote
// inside and be cut out. A space's vote pools many points, and a pocket that opens to the air
// always has points whose rays see out.
struct Spaces {
    int n = 0;              // cells per side, with a margin
    float lo = 0, step = 0; // world position of cell 0's corner, and the cell size
    std::vector<uint8_t> state;  // 0 empty, 1 wall, 2 outside, 3 enclosed and inside

    int64_t at(int x, int y, int z) const { return ((int64_t)x * n + y) * n + z; }
    int cell_of(float v) const { return std::min(n - 1, std::max(0, (int)std::floor((v - lo) / step))); }

    static Spaces build(const Mesh& mesh, const TriBvh& bvh, int res, float scale) {
        Spaces s;
        const int margin = SPACE_CLOSE + 2;
        const int g = std::max(32, res / SPACE_COARSEN);
        s.step = scale / (float)g;
        s.n = g + 2 * margin;
        s.lo = -0.5f * scale - margin * s.step;
        const int64_t N = (int64_t)s.n * s.n * s.n;
        std::vector<uint8_t> wall((size_t)N, 0);
        // The surface's faces are about a band cell across, a quarter of a coarse cell, so their
        // corners and centres mark every coarse cell they pass through.
        const int64_t F = (int64_t)mesh.faces.size() / 3;
        for (int64_t f = 0; f < F; ++f) {
            float c[3] = {0, 0, 0};
            for (int j = 0; j < 3; ++j) {
                const float* p = &mesh.verts[3 * (size_t)mesh.faces[3*f+j]];
                wall[s.at(s.cell_of(p[0]), s.cell_of(p[1]), s.cell_of(p[2]))] = 1;
                for (int k = 0; k < 3; ++k) c[k] += p[k] / 3.f;
            }
            wall[s.at(s.cell_of(c[0]), s.cell_of(c[1]), s.cell_of(c[2]))] = 1;
        }
        auto dilate = [&s](std::vector<uint8_t>& m, int r) {
            for (int axis = 0; axis < 3; ++axis) {
                std::vector<uint8_t> out = m;
                const int64_t stride = axis == 0 ? (int64_t)s.n * s.n : axis == 1 ? s.n : 1;
                for (int x = 0; x < s.n; ++x) for (int y = 0; y < s.n; ++y) for (int z = 0; z < s.n; ++z) {
                    const int64_t i = s.at(x, y, z);
                    if (!m[i]) continue;
                    const int c = axis == 0 ? x : axis == 1 ? y : z;
                    for (int d = 1; d <= r; ++d) {
                        if (c + d < s.n) out[i + d * stride] = 1;
                        if (c - d >= 0) out[i - d * stride] = 1;
                    }
                }
                m.swap(out);
            }
        };
        auto flood = [&s](std::vector<uint8_t>& reached, const std::vector<uint8_t>& blocked,
                          int64_t seed, std::vector<int64_t>* members) {
            std::vector<int64_t> stack(1, seed);
            reached[seed] = 1;
            while (!stack.empty()) {
                const int64_t i = stack.back(); stack.pop_back();
                if (members) members->push_back(i);
                const int x = (int)(i / ((int64_t)s.n * s.n)), y = (int)((i / s.n) % s.n), z = (int)(i % s.n);
                const int nb[6][3] = {{x-1,y,z},{x+1,y,z},{x,y-1,z},{x,y+1,z},{x,y,z-1},{x,y,z+1}};
                for (const auto& q : nb) {
                    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= s.n || q[1] >= s.n || q[2] >= s.n) continue;
                    const int64_t j = s.at(q[0], q[1], q[2]);
                    if (reached[j] || blocked[j]) continue;
                    reached[j] = 1;
                    stack.push_back(j);
                }
            }
        };
        // The outside: flooded from a corner past walls grown by SPACE_CLOSE, then grown back by as
        // much, which gives back the air beside the surface but not what lies behind a narrow
        // opening.
        std::vector<uint8_t> outside((size_t)N, 0);
        {
            std::vector<uint8_t> grown = wall;
            dilate(grown, SPACE_CLOSE);
            flood(outside, grown, 0, nullptr);
            dilate(outside, SPACE_CLOSE);
        }
        s.state.assign((size_t)N, 0);
        for (int64_t i = 0; i < N; ++i) s.state[i] = wall[i] ? 1 : outside[i] ? 2 : 0;
        // Cells with a clear cell between them and every wall. A crease the closing sealed off
        // (an armpit, the gap under a lapel) is a cell or two across and has none; a hollow does.
        std::vector<uint8_t> near_wall = wall;
        dilate(near_wall, 1);
        // Each enclosed space, voted as a whole.
        std::vector<uint8_t> seen((size_t)N, 0);
        std::vector<int64_t> members;
        int spaces = 0, inside_spaces = 0;
        for (int64_t i = 0; i < N; ++i) {
            if (s.state[i] != 0 || seen[i]) continue;
            members.clear();
            flood(seen, s.state, i, &members);  // walls and outside are non-zero: they block
            ++spaces;
            // Only a space with room in it can be a hollow, and its deep cells, clear of the
            // doubled sheets along its walls, are the ones voted.
            std::vector<int64_t> deep;
            for (int64_t c : members)
                if (!near_wall[c]) deep.push_back(c);
            if (deep.size() < (size_t)SPACE_DEEP) continue;
            const size_t m = deep.size();
            const int want = (int)std::min<size_t>(SPACE_SAMPLES, m);
            std::atomic<int> yes{0};
            parallel_for(want, [&](int64_t b, int64_t e) {
                for (int64_t k = b; k < e; ++k) {
                    const int64_t c = deep[want > 1 ? (size_t)k * (m - 1) / (size_t)(want - 1) : 0];
                    const int x = (int)(c / ((int64_t)s.n * s.n)), y = (int)((c / s.n) % s.n), z = (int)(c % s.n);
                    const float p[3] = {s.lo + (x + 0.5f) * s.step, s.lo + (y + 0.5f) * s.step, s.lo + (z + 0.5f) * s.step};
                    if (vote_inside(bvh, p)) yes.fetch_add(1, std::memory_order_relaxed);
                }
            });
            if ((float)yes.load() >= SPACE_INSIDE * (float)want) {
                for (int64_t c : members) s.state[c] = 3;
                ++inside_spaces;
            }
        }
        printf("  remesh_dc: fill_inside: coarse grid %d^3, %d enclosed spaces, %d inside\n", s.n, spaces, inside_spaces);
        fflush(stdout);
        return s;
    }

    // Whether the first empty coarse cell in front of `p` along the unit normal `n` lies in an
    // enclosed space voted inside. The search stops two cells out, once past the face's own wall:
    // in a crease (an armpit) the cells ahead stay walls, and going on would cross the crease into
    // the hollow of the part across it. A face with only walls ahead is kept.
    bool inside_ahead(const float p[3], const float n[3]) const {
        for (int k = 1; k <= 4; ++k) {
            const float d = 0.5f * step * (float)k;
            const uint8_t st = state[at(cell_of(p[0] + n[0] * d), cell_of(p[1] + n[1] * d), cell_of(p[2] + n[2] * d))];
            if (st == 1) continue;
            return st == 3;
        }
        return false;
    }
};

// Removes the band's inner boundary. The dual-contoured band is a closed surface wound outwards
// from the band, so each face looks out of the wall it bounds: the outer skin looks out of the
// solid, and the inner skin, a band width inside it, looks into what the input encloses. The faces
// that look into an enclosed space voted inside (`Spaces`) are the inner skin. Openings in the
// input (an eye socket open into a hollow body) join the two skins at their rims, so removing the
// inner skin leaves a hole there, and each such hole is capped: the surface closes over the
// opening.
void drop_inner_skin(Mesh& mesh, const TriBvh& bvh, int res, float scale) {
    const int64_t F = (int64_t)mesh.faces.size() / 3;
    if (F == 0) return;
    const Spaces spaces = Spaces::build(mesh, bvh, res, scale);
    std::vector<uint8_t> inner((size_t)F, 0);
    parallel_for(F, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f) {
            const float* a = &mesh.verts[3 * (size_t)mesh.faces[3*f]];
            const float* v = &mesh.verts[3 * (size_t)mesh.faces[3*f+1]];
            const float* c = &mesh.verts[3 * (size_t)mesh.faces[3*f+2]];
            const float e1[3] = {v[0]-a[0], v[1]-a[1], v[2]-a[2]};
            const float e2[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
            float n[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
            const float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (len <= 0.f) continue;
            for (int k = 0; k < 3; ++k) n[k] /= len;
            const float centre[3] = {(a[0]+v[0]+c[0]) / 3.f, (a[1]+v[1]+c[1]) / 3.f, (a[2]+v[2]+c[2]) / 3.f};
            inner[f] = spaces.inside_ahead(centre, n) ? 1 : 0;
        }
    });

    const EdgeUse use = EdgeUse::of(mesh.faces);
    // Patches of one label smaller than NOISE_FACES take the other: inner specks on the outer skin
    // first, then outer specks left inside a hollow.
    std::vector<int32_t> seen((size_t)F, -1), stack, patch;
    int64_t flipped = 0;
    for (uint8_t label : {uint8_t(1), uint8_t(0)}) {
        std::fill(seen.begin(), seen.end(), -1);
        for (int64_t f0 = 0; f0 < F; ++f0) {
            if (inner[f0] != label || seen[f0] >= 0) continue;
            patch.clear();
            stack.assign(1, (int32_t)f0);
            seen[f0] = 1;
            while (!stack.empty()) {
                const int32_t f = stack.back(); stack.pop_back();
                patch.push_back(f);
                for (int k = 0; k < 3; ++k) {
                    const int64_t t = use.twin[(size_t)(3 * (int64_t)f + k)];
                    const int32_t g = t < 0 ? -1 : (int32_t)(t / 3);
                    if (g < 0 || inner[g] != label || seen[g] >= 0) continue;
                    seen[g] = 1;
                    stack.push_back(g);
                }
            }
            if (patch.size() > NOISE_FACES) continue;
            for (int32_t f : patch) inner[f] = 1 - label;
            flipped += (int64_t)patch.size();
        }
    }
    // Where two sheets of the band touch, an edge carries four faces, the two skins of each sheet.
    // Removing the inner ones would leave the two outer faces on one edge wound the same way,
    // which the later winding repair reads as a fault and answers by turning a whole sheet inside
    // out. Such edges keep all their faces; the few inner ones kept stay hidden inside.
    int64_t pinned = 0;
    for (int64_t f = 0; f < F; ++f) {
        if (!inner[f]) continue;
        for (int k = 0; k < 3; ++k)
            if (use.count[(size_t)(3 * f + k)] > 2) { inner[f] = 0; ++pinned; break; }
    }

    std::vector<int32_t> kept;
    kept.reserve(mesh.faces.size());
    int64_t dropped = 0;
    for (int64_t f = 0; f < F; ++f) {
        if (inner[f]) { ++dropped; continue; }
        kept.insert(kept.end(), &mesh.faces[3*f], &mesh.faces[3*f] + 3);
    }
    mesh.faces.swap(kept);
    int capped = 0;
    int64_t hollow = 0;
    if (dropped) {
        // The band surface is watertight, so every loop now open is the rim of a removed patch.
        capped = cap_loops(mesh);
        hollow = drop_hollows(mesh);
        std::vector<int32_t> remap(mesh.verts.size() / 3, -1);
        std::vector<float> verts;
        for (int32_t& v : mesh.faces) {
            if (remap[v] < 0) {
                remap[v] = (int32_t)(verts.size() / 3);
                verts.insert(verts.end(), &mesh.verts[3 * (size_t)v], &mesh.verts[3 * (size_t)v] + 3);
            }
            v = remap[v];
        }
        mesh.verts.swap(verts);
    }
    printf("  remesh_dc: fill_inside: dropped %lld inner-skin faces (%lld relabelled as noise, %lld kept on shared "
           "edges), capped %d openings, dropped %lld faces of sealed hollows\n",
           (long long)dropped, (long long)flipped, (long long)pinned, capped, (long long)hollow);
    fflush(stdout);
}

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back,
                           bool fill_inside) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    auto bit_set = [&cand, res](int x, int y, int z) {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        cand[(size_t)(i >> 6)] |= 1ull << (i & 63);
    };
    auto bit_get = [&cand, res](int x, int y, int z) -> bool {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return (cand[(size_t)(i >> 6)] >> (i & 63)) & 1;
    };
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
        // Each worker previously allocated a full res^3 candidate bitset.  At
        // res=1024 that is 128 MiB per worker, so a 32-thread CPU consumed ~4 GiB
        // here before any geometry/BVH memory.  Cap only the replication memory;
        // the OR result and therefore the remesh are bit-for-bit equivalent.
        const size_t bytes_per_part = cand.size() * sizeof(uint64_t);
        const size_t parts_budget = (size_t)1024 * 1024 * 1024; // 1 GiB
        const int mem_workers = bytes_per_part ? (int)std::max<size_t>(1, parts_budget / bytes_per_part) : hw;
        const int nt = std::max(1, std::min(hw, mem_workers));
        if (nt < hw) {
            printf("  [remesh-mem] candidate bitset %.1f MiB/worker, workers %d->%d\n",
                   bytes_per_part / (1024.0*1024.0), hw, nt);
            fflush(stdout);
        }
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            ts.emplace_back([&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        std::vector<int64_t> cand_cells;
        for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                const int b = ctz64(bits);
                bits &= bits - 1;
                cand_cells.push_back((w << 6) | b);
            }
        }
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - eps;
                if (std::fabs(f) < keep) act[i] = 1;
            }
        });
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            if (!act[i]) continue;
            const int64_t c = cand_cells[i];
            acoord.push_back((int)(c / ((int64_t)res * res)));
            acoord.push_back((int)((c / res) % res));
            acoord.push_back((int)(c % res));
        }
        cand.clear(); cand.shrink_to_fit();
    }
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
    std::vector<int> vcoord;
    std::unordered_map<uint64_t, int> vmap;
    vmap.reserve((size_t)Na * 3);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
            const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
            if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
            }
        }
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            fvert[i] = (h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell) - eps;
        }
    });
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[it->second];
    };

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    a0[(axis + 1) % 3] += u;
                    a0[(axis + 2) % 3] += v;
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const float v1 = fval(a0[0], a0[1], a0[2]);
                    const float v2 = fval(a1[0], a1[1], a1[2]);
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                auto it = vox.find(key3(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]));
                if (it == vox.end()) ok = false;
                else q[k] = it->second;
            }
            if (!ok) continue;
            static const int S1N[6] = {0, 1, 2, 0, 2, 3};
            static const int S1P[6] = {0, 2, 1, 0, 3, 2};
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];
    if (fill_inside) drop_inner_skin(out, bvh, res, scale);

    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
