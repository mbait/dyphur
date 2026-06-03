#include <core/broadphase.hpp>
#include <compute/atomics.hpp>
#include <compute/kernel.hpp>
#include <compute/sort.hpp>
#include <sycl/sycl.hpp>
#include <cassert>

namespace dyphur {

// ── Device helpers ────────────────────────────────────────────────────────────

namespace {

inline uint32_t next_pow2(uint32_t n) {
    if (n <= 1u) return 1u;
    --n;
    n |= n >> 1u; n |= n >> 2u; n |= n >> 4u; n |= n >> 8u; n |= n >> 16u;
    return n + 1u;
}

// Spread 10-bit integer to 30 bits by inserting 2 zeros between each bit.
inline uint32_t expand_bits(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16u)) & 0xFF0000FFu;
    v = (v | (v <<  8u)) & 0x0F00F00Fu;
    v = (v | (v <<  4u)) & 0xC30C30C3u;
    v = (v | (v <<  2u)) & 0x49249249u;
    return v;
}

inline uint32_t morton3(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1u) | (expand_bits(z) << 2u);
}

// Karras 2012: longest-common-prefix length between sorted positions i and j.
// Keys are (morton30 << 32 | body_idx) — always distinct, no equal-code case.
// Returns -1 for out-of-bounds queries.
inline int lbvh_delta(int i, int j, const uint64_t* codes, int n) {
    if (j < 0 || j >= n) return -1;
    return static_cast<int>(sycl::clz(codes[i] ^ codes[j]));
}

// Karras: find the leaf range [lo, hi] covered by internal node i.
inline void find_range(int i, const uint64_t* codes, int n,
                       int& lo, int& hi) {
    int d = (lbvh_delta(i, i + 1, codes, n) - lbvh_delta(i, i - 1, codes, n)) >= 0 ? 1 : -1;
    int delta_min = lbvh_delta(i, i - d, codes, n);

    int l_max = 2;
    while (lbvh_delta(i, i + l_max * d, codes, n) > delta_min)
        l_max <<= 1;

    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (lbvh_delta(i, i + (l + t) * d, codes, n) > delta_min)
            l += t;
    }

    int j = i + l * d;
    lo = d > 0 ? i : j;
    hi = d > 0 ? j : i;
}

// Find the Morton-code split point within [lo, hi].
// Returns gamma in [lo, hi-1] such that range [lo, gamma] goes left.
inline int find_split(int lo, int hi, const uint64_t* codes, int n) {
    int delta_node = lbvh_delta(lo, hi, codes, n);
    int s    = 0;
    int step = hi - lo;
    do {
        step = (step + 1) >> 1;
        if (lbvh_delta(lo, lo + s + step, codes, n) > delta_node)
            s += step;
    } while (step > 1);
    return lo + s;
}

} // anonymous namespace

// ── Constructor ───────────────────────────────────────────────────────────────

Broadphase::Broadphase(Stream& s, uint32_t max_bodies, uint32_t max_pairs)
    : max_bodies_(max_bodies)
    , max_padded_(next_pow2(max_bodies))
    , max_pairs_(max_pairs)
    , d_morton_(s, max_padded_)
    , d_sorted_idx_(s, max_padded_)
    , d_left_(s, max_bodies > 1 ? max_bodies - 1 : 1)
    , d_right_(s, max_bodies > 1 ? max_bodies - 1 : 1)
    , d_parent_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_min_x_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_min_y_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_min_z_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_max_x_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_max_y_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_aabb_max_z_(s, max_bodies > 1 ? 2 * max_bodies - 1 : 1)
    , d_flags_(s, max_bodies > 1 ? max_bodies - 1 : 1)
    , d_root_(s, 1)
    , d_pairs_(s, max_pairs)
    , d_count_(s, 1)
    , d_pair_keys_(s, max_pairs)
{}

// ── Build + Query ─────────────────────────────────────────────────────────────

void Broadphase::build_and_query(Stream&         s,
                                 const BodyView&  bodies,
                                 const ShapeView& shapes,
                                 const AABB&      scene_bounds) {
    const uint32_t n = bodies.n;
    assert(n <= max_bodies_);

    // Zero the pair counter regardless of n so the result is always valid.
    auto& q = s.queue();
    q.memset(d_count_.data(), 0, sizeof(uint32_t));

    if (n < 2) return;

    const int     n_int    = static_cast<int>(n);
    const uint32_t n_pad   = max_padded_;

    // ── Step 1: Morton codes + identity permutation ──────────────────────────
    {
        float sx0 = scene_bounds.min_x, sx1 = scene_bounds.max_x;
        float sy0 = scene_bounds.min_y, sy1 = scene_bounds.max_y;
        float sz0 = scene_bounds.min_z, sz1 = scene_bounds.max_z;
        uint64_t* d_mc  = d_morton_.data();
        uint32_t* d_si  = d_sorted_idx_.data();
        const float* px = bodies.pos_x;
        const float* py = bodies.pos_y;
        const float* pz = bodies.pos_z;

        parallel_for(s, n_pad, [=](size_t i) {
            if (i < static_cast<size_t>(n_int)) {
                float tx = (px[i] - sx0) / (sx1 - sx0);
                float ty = (py[i] - sy0) / (sy1 - sy0);
                float tz = (pz[i] - sz0) / (sz1 - sz0);
                tx = tx < 0.f ? 0.f : (tx > 1.f ? 1.f : tx);
                ty = ty < 0.f ? 0.f : (ty > 1.f ? 1.f : ty);
                tz = tz < 0.f ? 0.f : (tz > 1.f ? 1.f : tz);
                uint32_t qx = static_cast<uint32_t>(tx * 1023.f);
                uint32_t qy = static_cast<uint32_t>(ty * 1023.f);
                uint32_t qz = static_cast<uint32_t>(tz * 1023.f);
                // Embed body index in low 32 bits: stable sort for equal Morton codes.
                d_mc[i] = ((uint64_t)morton3(qx, qy, qz) << 32) | (uint64_t)i;
                d_si[i] = static_cast<uint32_t>(i);
            } else {
                d_mc[i] = ~0ULL;  // pads sort to the end
                d_si[i] = ~0u;
            }
        });
    }

    // ── Step 2: Sort by Morton code ──────────────────────────────────────────
    sort_by_key(s, d_morton_.data(), d_sorted_idx_.data(), static_cast<size_t>(n_pad));

    // ── Step 3: Initialise node metadata ─────────────────────────────────────
    // parent = -1 (0xFF bytes), flags = 0.
    q.memset(d_parent_.data(), 0xFF, static_cast<size_t>(2 * n - 1) * sizeof(int32_t));
    q.memset(d_flags_.data(),  0,    static_cast<size_t>(n - 1) * sizeof(uint32_t));

    // ── Step 4: Karras tree construction ─────────────────────────────────────
    {
        const uint64_t* d_codes = d_morton_.data();
        int32_t* d_left   = d_left_.data();
        int32_t* d_right  = d_right_.data();
        int32_t* d_parent = d_parent_.data();

        parallel_for(s, static_cast<size_t>(n - 1), [=](size_t idx) {
            int i = static_cast<int>(idx);
            int lo, hi;
            find_range(i, d_codes, n_int, lo, hi);
            int gamma = find_split(lo, hi, d_codes, n_int);

            // Encode children as flat node indices.
            // Left child covers [lo, gamma]:
            int left  = (lo == gamma)     ? (n_int - 1 + lo)          : gamma;
            // Right child covers [gamma+1, hi]:
            int right = (gamma + 1 == hi) ? (n_int - 1 + gamma + 1) : (gamma + 1);

            d_left[i]  = left;
            d_right[i] = right;
            d_parent[left]  = i;
            d_parent[right] = i;
        });
    }

    // ── Step 5: Find root (internal node whose parent stayed -1) ─────────────
    {
        const int32_t* d_parent = d_parent_.data();
        int32_t*       d_root   = d_root_.data();

        parallel_for(s, static_cast<size_t>(n - 1), [=](size_t i) {
            if (d_parent[i] == -1)
                d_root[0] = static_cast<int32_t>(i);
        });
    }

    // ── Step 6: Refit — bottom-up AABB propagation (level-synchronised) ───────
    // A parallel refit must not have one thread read a child AABB another thread
    // is still writing — CUDA's device-scope atomics don't reliably order the
    // non-atomic fp16 AABB writes against the rendezvous flag, which made the old
    // atomic-walk-up refit non-deterministic at scale (known_issues Issue 4).
    // Instead: (a) a parallel pass computes every leaf AABB; (b) repeated rounds,
    // each a *separate* queue-ordered kernel, merge an internal node once both its
    // children are finalised. Because a parent always reads child AABBs that were
    // written by a *previous* kernel, the reads are ordered and the result is
    // bit-identical every run. `d_flags_` (zeroed in Step 3) doubles as the
    // per-internal-node "finalised" flag; leaves (index ≥ n_int-1) are always ready.
    {
        const uint32_t* d_si   = d_sorted_idx_.data();
        const int32_t*  d_lft  = d_left_.data();
        const int32_t*  d_rgt  = d_right_.data();
        sycl::half* d_mn_x = d_aabb_min_x_.data();
        sycl::half* d_mn_y = d_aabb_min_y_.data();
        sycl::half* d_mn_z = d_aabb_min_z_.data();
        sycl::half* d_mx_x = d_aabb_max_x_.data();
        sycl::half* d_mx_y = d_aabb_max_y_.data();
        sycl::half* d_mx_z = d_aabb_max_z_.data();
        uint32_t* d_flags = d_flags_.data();
        const BodyView  bv = bodies;
        const ShapeView sv = shapes;
        const int ni = n_int;

        // (a) Parallel leaf AABBs.
        parallel_for(s, static_cast<size_t>(n_int), [=](size_t leaf_k) {
            uint32_t body = d_si[leaf_k];
            float px = bv.pos_x[body], py = bv.pos_y[body], pz = bv.pos_z[body];
            uint32_t sh = bv.shape[body], stype = sv.type[sh];
            float hx = sv.half_x[sh];
            float mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
            if (stype == static_cast<uint32_t>(ShapeType::Sphere)) {
                mn_x = px - hx; mx_x = px + hx;
                mn_y = py - hx; mx_y = py + hx;
                mn_z = pz - hx; mx_z = pz + hx;
            } else { // Box
                float qw = bv.rot_w[body], qx = bv.rot_x[body], qy = bv.rot_y[body], qz = bv.rot_z[body];
                float hy = sv.half_y[sh], hz = sv.half_z[sh];
                float rxx = 1.f - 2.f*(qy*qy + qz*qz);
                float rxy =        2.f*(qx*qy - qz*qw);
                float rxz =        2.f*(qx*qz + qy*qw);
                float ryx =        2.f*(qx*qy + qz*qw);
                float ryy = 1.f - 2.f*(qx*qx + qz*qz);
                float ryz =        2.f*(qy*qz - qx*qw);
                float rzx =        2.f*(qx*qz - qy*qw);
                float rzy =        2.f*(qy*qz + qx*qw);
                float rzz = 1.f - 2.f*(qx*qx + qy*qy);
                float ex = sycl::fabs(rxx)*hx + sycl::fabs(rxy)*hy + sycl::fabs(rxz)*hz;
                float ey = sycl::fabs(ryx)*hx + sycl::fabs(ryy)*hy + sycl::fabs(ryz)*hz;
                float ez = sycl::fabs(rzx)*hx + sycl::fabs(rzy)*hy + sycl::fabs(rzz)*hz;
                mn_x = px - ex; mx_x = px + ex;
                mn_y = py - ey; mx_y = py + ey;
                mn_z = pz - ez; mx_z = pz + ez;
            }
            int nidx = ni - 1 + static_cast<int>(leaf_k);
            d_mn_x[nidx] = sycl::half(mn_x);  d_mx_x[nidx] = sycl::half(mx_x);
            d_mn_y[nidx] = sycl::half(mn_y);  d_mx_y[nidx] = sycl::half(mx_y);
            d_mn_z[nidx] = sycl::half(mn_z);  d_mx_z[nidx] = sycl::half(mx_z);
        });

        // (b) Level-synchronised merge rounds. One kernel per round; a node merges
        // only once both children are finalised (children come from prior kernels).
        // The Karras tree over distinct 64-bit keys has height ≤ 64, so ≤ 64 rounds
        // always complete it; we run in batches and stop once the root is finalised.
        if (n_int > 1) {
            int32_t root_h;
            q.memcpy(&root_h, d_root_.data(), sizeof(int32_t)).wait();
            const size_t n_internal = static_cast<size_t>(n_int - 1);
            constexpr int BATCH = 16, MAX_BATCH = 4;   // 64 rounds max
            for (int batch = 0; batch < MAX_BATCH; ++batch) {
                for (int r = 0; r < BATCH; ++r) {
                    parallel_for(s, n_internal, [=](size_t pp) {
                        int p = static_cast<int>(pp);
                        if (d_flags[p]) return;                 // already finalised
                        int lc = d_lft[p], rc = d_rgt[p];
                        bool lready = (lc >= ni - 1) || d_flags[lc];
                        bool rready = (rc >= ni - 1) || d_flags[rc];
                        if (!lready || !rready) return;
                        float pmnx = sycl::fmin(float(d_mn_x[lc]), float(d_mn_x[rc]));
                        float pmny = sycl::fmin(float(d_mn_y[lc]), float(d_mn_y[rc]));
                        float pmnz = sycl::fmin(float(d_mn_z[lc]), float(d_mn_z[rc]));
                        float pmxx = sycl::fmax(float(d_mx_x[lc]), float(d_mx_x[rc]));
                        float pmxy = sycl::fmax(float(d_mx_y[lc]), float(d_mx_y[rc]));
                        float pmxz = sycl::fmax(float(d_mx_z[lc]), float(d_mx_z[rc]));
                        d_mn_x[p] = sycl::half(pmnx);  d_mx_x[p] = sycl::half(pmxx);
                        d_mn_y[p] = sycl::half(pmny);  d_mx_y[p] = sycl::half(pmxy);
                        d_mn_z[p] = sycl::half(pmnz);  d_mx_z[p] = sycl::half(pmxz);
                        d_flags[p] = 1u;
                    });
                }
                uint32_t root_ready = 0;
                q.memcpy(&root_ready, d_flags + root_h, sizeof(uint32_t)).wait();
                if (root_ready) break;
            }
        }
    }

    // ── Step 7: Traversal — emit overlapping candidate pairs ─────────────────
    // Each leaf j queries the BVH from the root; emits pair (leaf_i, j) when
    // leaf_i < j and their AABBs overlap. Canonical body ordering (a < b) is
    // enforced inside the pair before writing.
    {
        const int32_t*    d_root  = d_root_.data();
        const int32_t*    d_lft   = d_left_.data();
        const int32_t*    d_rgt   = d_right_.data();
        const sycl::half* d_mn_x  = d_aabb_min_x_.data();
        const sycl::half* d_mn_y  = d_aabb_min_y_.data();
        const sycl::half* d_mn_z  = d_aabb_min_z_.data();
        const sycl::half* d_mx_x  = d_aabb_max_x_.data();
        const sycl::half* d_mx_y  = d_aabb_max_y_.data();
        const sycl::half* d_mx_z  = d_aabb_max_z_.data();
        const uint32_t*   d_si    = d_sorted_idx_.data();
        ContactPair*      d_pairs = d_pairs_.data();
        uint32_t*         d_cnt   = d_count_.data();
        const uint32_t    cap     = max_pairs_;

        parallel_for(s, static_cast<size_t>(n), [=](size_t leaf_j) {
            int jnode = n_int - 1 + static_cast<int>(leaf_j);
            // Load leaf AABB once as fp32 for repeated comparisons.
            float jmnx = float(d_mn_x[jnode]), jmxx = float(d_mx_x[jnode]);
            float jmny = float(d_mn_y[jnode]), jmxy = float(d_mx_y[jnode]);
            float jmnz = float(d_mn_z[jnode]), jmxz = float(d_mx_z[jnode]);

            int  root = d_root[0];
            int  stack[64];
            int  top = 0;
            stack[top++] = root;

            while (top > 0) {
                int node = stack[--top];

                // AABB overlap test: promote fp16 node bounds to fp32.
                if (float(d_mn_x[node]) > jmxx || float(d_mx_x[node]) < jmnx ||
                    float(d_mn_y[node]) > jmxy || float(d_mx_y[node]) < jmny ||
                    float(d_mn_z[node]) > jmxz || float(d_mx_z[node]) < jmnz)
                    continue;

                bool is_leaf = (node >= n_int - 1);
                if (is_leaf) {
                    int leaf_i = node - (n_int - 1);
                    if (static_cast<size_t>(leaf_i) < leaf_j) {
                        uint32_t idx = atomic_add_seq(d_cnt, 1u);
                        if (idx < cap) {
                            uint32_t a = d_si[leaf_i], b = d_si[leaf_j];
                            if (a > b) { uint32_t t = a; a = b; b = t; }
                            d_pairs[idx] = { a, b };
                        }
                    }
                    continue;
                }
                // Internal node: push both children.
                stack[top++] = d_lft[node];
                stack[top++] = d_rgt[node];
            }
        });
    }
}

void Broadphase::sort_pairs(Stream& s, uint32_t n_pairs) {
    if (n_pairs <= 1) return;
    uint32_t n_sort = next_pow2(n_pairs);
    // Init sort region to UINT64_MAX so padding goes to the end.
    s.queue().memset(d_pair_keys_.data(), 0xFF, n_sort * sizeof(uint64_t));
    const ContactPair* pairs = d_pairs_.data();
    uint64_t* keys = d_pair_keys_.data();
    parallel_for(s, n_pairs, [pairs, keys](size_t i) {
        keys[i] = ((uint64_t)pairs[i].a << 32) | (uint64_t)pairs[i].b;
    });
    sort_by_key(s, keys, d_pairs_.data(), static_cast<size_t>(n_sort));
}

uint32_t Broadphase::download_count(Stream& s) const {
    uint32_t cnt;
    s.queue().memcpy(&cnt, d_count_.data(), sizeof(cnt)).wait();
    return cnt;
}

BvhView Broadphase::bvh_view(uint32_t n_active) const noexcept {
    return BvhView{
        d_left_.data(),
        d_right_.data(),
        d_parent_.data(),
        d_root_.data(),
        d_sorted_idx_.data(),
        d_aabb_min_x_.data(),
        d_aabb_min_y_.data(),
        d_aabb_min_z_.data(),
        d_aabb_max_x_.data(),
        d_aabb_max_y_.data(),
        d_aabb_max_z_.data(),
        n_active,
    };
}

} // namespace dyphur
