// BiRefNet (Swin-L backbone + ASPPDeformable decoder) in GGML. The backbone runs as one graph per
// input scale; window partition / shifted-window roll / attention mask are precomputed on the host
// as gather/scatter index arrays + an additive mask, applied via ggml_get_rows. Each decoder block
// runs as two graphs around its deformable convolutions, the one step done by a custom kernel.
// Validated against tools/ref_birefnet.py dumps.
#include "birefnet.h"
#include "deform_conv.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <cstdio>

namespace trellis {
using T = ggml_tensor;

static const int WS = 12;          // window size (swin_v1_l)
static const int HEADS[4] = {6, 12, 24, 48};
static const int DEPTH[4] = {2, 2, 18, 2};

static ggml_context* mkctx(size_t nodes = 16384) {
    size_t meta = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// build + run a graph; returns each output read to host f32 (read BEFORE freeing the gallocr).
static std::vector<std::vector<float>> run_graph(const Model& m, ggml_context* c, std::vector<T*> outs,
                      std::vector<std::pair<T*, const void*>> ins, size_t nodes = 16384) {
    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    for (T* o : outs) { ggml_set_output(o); ggml_build_forward_expand(g, o); }
    ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(a, g)) throw std::runtime_error("birefnet alloc");
    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr, "      [birefnet-alloc] %-48s nodes=%d buffer=%.3f GB\n",
                outs.empty() ? "" : ggml_get_name(outs.front()), ggml_graph_n_nodes(g),
                ggml_gallocr_get_buffer_size(a, 0) / 1e9);
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("birefnet compute");
    std::vector<std::vector<float>> r;
    for (T* o : outs) r.push_back(tensor_to_f32(o));
    ggml_gallocr_free(a);
    return r;
}

static T* ln(ggml_context* c, T* x, T* w, T* b, float eps = 1e-5f) {  // LayerNorm over ne0
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    return ggml_add(c, x, b);
}
static T* lin(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* y = ggml_mul_mat(c, m.get(p + ".weight"), x);
    if (m.has(p + ".bias")) y = ggml_add(c, y, m.get(p + ".bias"));
    return y;
}

// ---- host window index bookkeeping ----
struct WinIdx {
    int Hp, Wp, nWh, nWw, nW, N;
    std::vector<int32_t> gather;   // [nW*WS*WS] padded-grid token -> real token (or N for zero pad)
    std::vector<int32_t> scatter;  // [N] real token -> window-major position
    std::vector<float>   mask;     // [WS*WS(k) * WS*WS(q) * nW] additive (shifted only), else empty
};
static int wrap(int a, int n) { return ((a % n) + n) % n; }

static WinIdx build_winidx(int H, int W, int shift) {
    WinIdx wi;
    wi.N = H * W;
    wi.Hp = ((H + WS - 1) / WS) * WS; wi.Wp = ((W + WS - 1) / WS) * WS;
    wi.nWh = wi.Hp / WS; wi.nWw = wi.Wp / WS; wi.nW = wi.nWh * wi.nWw;
    const int ws2 = WS * WS, N = wi.N;
    wi.gather.resize((size_t)wi.nW * ws2);
    for (int w = 0; w < wi.nW; ++w) {
        int wr = w / wi.nWw, wc = w % wi.nWw;
        for (int p = 0; p < ws2; ++p) {
            int pr = p / WS, pc = p % WS;
            int gh = wr * WS + pr, gw = wc * WS + pc;
            int oh = wrap(gh + shift, wi.Hp), ow = wrap(gw + shift, wi.Wp);
            wi.gather[(size_t)w * ws2 + p] = (oh < H && ow < W) ? (oh * W + ow) : N;
        }
    }
    wi.scatter.resize(N);
    for (int n = 0; n < N; ++n) {
        int oh = n / W, ow = n % W;
        int sh = wrap(oh - shift, wi.Hp), sw = wrap(ow - shift, wi.Wp);
        int wr = sh / WS, wc = sw / WS, pr = sh % WS, pc = sw % WS;
        wi.scatter[n] = (wr * wi.nWw + wc) * ws2 + (pr * WS + pc);
    }
    if (shift > 0) {
        std::vector<int> img((size_t)wi.Hp * wi.Wp);
        auto region = [&](int x, int len) { return x < len - WS ? 0 : (x < len - shift ? 1 : 2); };
        for (int gh = 0; gh < wi.Hp; ++gh) for (int gw = 0; gw < wi.Wp; ++gw)
            img[(size_t)gh * wi.Wp + gw] = region(gh, wi.Hp) * 3 + region(gw, wi.Wp);
        std::vector<int> mw((size_t)wi.nW * ws2);
        for (int w = 0; w < wi.nW; ++w) { int wr = w / wi.nWw, wc = w % wi.nWw;
            for (int p = 0; p < ws2; ++p) { int pr = p / WS, pc = p % WS;
                mw[(size_t)w * ws2 + p] = img[(size_t)(wr*WS+pr) * wi.Wp + (wc*WS+pc)]; } }
        wi.mask.assign((size_t)ws2 * ws2 * wi.nW, 0.0f);   // [k, q, nW]
        for (int w = 0; w < wi.nW; ++w) for (int q = 0; q < ws2; ++q) for (int k = 0; k < ws2; ++k)
            wi.mask[(size_t)k + ws2 * (q + (size_t)ws2 * w)] =
                (mw[(size_t)w*ws2 + q] != mw[(size_t)w*ws2 + k]) ? -100.0f : 0.0f;
    }
    return wi;
}

// Everything a stage's graph needs from the host, for one token grid.
struct StageIdx {
    int H = 0, W = 0, N = 0, nW[2] = {0, 0};
    std::vector<int32_t> gather[2], scatter[2];   // [0] unshifted, [1] shifted; a pad slot gathers token 0
    std::vector<float>   keep[2];                 // [nW*ws2] 1 for a real token, 0 for a pad slot
    std::vector<float>   shift;                   // [ws2(k), ws2(q), 1, nW] additive shifted-window mask
    std::vector<int32_t> merge[4];                // PatchMerging's four 2x2 gathers
};

static StageIdx stage_idx(int H, int W) {
    StageIdx si; si.H = H; si.W = W; si.N = H * W;
    for (int sh = 0; sh < 2; ++sh) {
        WinIdx wi = build_winidx(H, W, sh ? WS / 2 : 0);
        si.nW[sh] = wi.nW;
        si.keep[sh].resize(wi.gather.size());
        for (size_t i = 0; i < wi.gather.size(); ++i) {
            const bool pad = wi.gather[i] == wi.N;
            si.keep[sh][i] = pad ? 0.0f : 1.0f;
            if (pad) wi.gather[i] = 0;
        }
        si.gather[sh] = std::move(wi.gather);
        si.scatter[sh] = std::move(wi.scatter);
        if (sh) si.shift = std::move(wi.mask);
    }
    const int Hh = H / 2, Wh = W / 2;
    for (int q = 0; q < 4; ++q) si.merge[q].resize((size_t)Hh * Wh);
    for (int hr = 0; hr < Hh; ++hr) for (int wr = 0; wr < Wh; ++wr) {
        const int np = hr * Wh + wr;
        si.merge[0][np] = (2*hr)*W + (2*wr);   si.merge[1][np] = (2*hr+1)*W + (2*wr);
        si.merge[2][np] = (2*hr)*W + (2*wr+1); si.merge[3][np] = (2*hr+1)*W + (2*wr+1);
    }
    return si;
}

// One Swin block, [C,N] -> [C,N], inside the backbone's graph. `mask` is soft_max_ext's additive
// mask: the relative-position bias [ws2,ws2,NH] alone for an unshifted block, broadcast over the
// windows, or bias + shift mask [ws2,ws2,NH,nW] for a shifted one.
static T* swin_block(ggml_context* c, const Model& m, const std::string& p, T* x, int C, int NH, int nW,
                     T* gather, T* keep, T* scatter, T* mask) {
    const int ws2 = WS * WS, HD = C / NH, Mwin = nW * ws2;
    const float scale = 1.0f / std::sqrt((float)HD);
    T* h = ln(c, x, m.get(p + ".norm1.weight"), m.get(p + ".norm1.bias"));
    h = ggml_get_rows(c, h, gather);                                // [C, Mwin]
    h = ggml_mul(c, h, keep);                                       // pad slots -> zero tokens
    T* qkv = lin(c, m, p + ".attn.qkv", h);                         // [3C, Mwin]
    T* q = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], 0);
    T* k = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], (size_t)C * ggml_element_size(qkv));
    T* v = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], (size_t)2 * C * ggml_element_size(qkv));
    auto heads = [&](T* t) {
        t = ggml_cont(c, t);
        t = ggml_reshape_4d(c, t, HD, NH, ws2, nW);
        t = ggml_cont(c, ggml_permute(c, t, 0, 2, 1, 3));
        return ggml_reshape_3d(c, t, HD, ws2, NH * nW);
    };
    q = heads(q); k = heads(k); v = heads(v);
    T* kq = ggml_mul_mat(c, k, q);                                  // [ws2(k), ws2(q), NH*nW]
    kq = ggml_reshape_4d(c, kq, ws2, ws2, NH, nW);
    kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);
    kq = ggml_reshape_3d(c, kq, ws2, ws2, NH * nW);
    T* vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));           // [ws2(k), HD, NH*nW]
    T* o = ggml_mul_mat(c, vt, kq);                                 // [HD, ws2(q), NH*nW]
    o = ggml_reshape_4d(c, o, HD, ws2, NH, nW);
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));
    o = ggml_reshape_2d(c, o, C, Mwin);
    o = lin(c, m, p + ".attn.proj", o);
    o = ggml_get_rows(c, o, scatter);                               // [C, N]
    T* xr = ggml_add(c, x, o);
    T* h2 = ln(c, xr, m.get(p + ".norm2.weight"), m.get(p + ".norm2.bias"));
    h2 = lin(c, m, p + ".mlp.fc1", h2);
    h2 = ggml_gelu_erf(c, h2);
    h2 = lin(c, m, p + ".mlp.fc2", h2);
    return ggml_add(c, xr, h2);
}

// The whole backbone at one scale as a single graph: patch embed, the four stages and their
// PatchMerging, with the tokens never leaving the device. The window bookkeeping comes from
// stage_idx and the relative-position bias is gathered on the device from the model's own table
// and index tensors. Each stage's normalized output is read back already in torch [C,H,W] order.
BBOut swin_backbone(const Model& m, const std::vector<float>& chw, int S) {
    const int ws2 = WS * WS;
    ggml_context* c = mkctx();
    std::vector<std::pair<T*, const void*>> ins;
    auto input = [&](T* t, const void* data) { ggml_set_input(t); ins.push_back({t, data}); return t; };

    T* img = input(ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1), chw.data());
    T* pe = ggml_conv_2d(c, m.get("bb.patch_embed.proj.weight"), img, 4, 4, 0, 0, 1, 1);   // [W,H,192]
    pe = ggml_add(c, pe, ggml_reshape_3d(c, m.get("bb.patch_embed.proj.bias"), 1, 1, 192));
    int C = 192, H = S / 4, W = S / 4;
    T* tok = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, pe, 1, 2, 0, 3)), C, W * H);  // [192, N]
    tok = ln(c, tok, m.get("bb.patch_embed.norm.weight"), m.get("bb.patch_embed.norm.bias"));

    BBOut out;
    T* outs[4];
    StageIdx idx[4];                     // host buffers the graph uploads; they live until it has run
    for (int s = 0; s < 4; ++s) {
        const int NH = HEADS[s];
        idx[s] = stage_idx(H, W);
        const StageIdx& si = idx[s];
        const std::string p0 = "bb.layers." + std::to_string(s) + ".blocks.";
        T* gather[2]; T* keep[2]; T* scatter[2];
        for (int sh = 0; sh < 2; ++sh) {
            const int Mwin = si.nW[sh] * ws2;
            gather[sh]  = input(ggml_new_tensor_1d(c, GGML_TYPE_I32, Mwin), si.gather[sh].data());
            keep[sh]    = input(ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, Mwin), si.keep[sh].data());
            scatter[sh] = input(ggml_new_tensor_1d(c, GGML_TYPE_I32, si.N), si.scatter[sh].data());
        }
        T* shift = input(ggml_new_tensor_4d(c, GGML_TYPE_F32, ws2, ws2, 1, si.nW[1]), si.shift.data());
        T* rel_index = ggml_reshape_1d(c, m.get(p0 + "0.attn.relative_position_index"), ws2 * ws2);  // k + ws2*q
        for (int b = 0; b < DEPTH[s]; ++b) {
            const std::string p = p0 + std::to_string(b);
            const int sh = b % 2;
            // bias[k,q,h] = table[h, index[k,q]]
            T* bias = ggml_get_rows(c, m.get(p + ".attn.relative_position_bias_table"), rel_index);  // [NH, k + ws2*q]
            bias = ggml_reshape_3d(c, bias, NH, ws2, ws2);                                           // [NH, k, q]
            bias = ggml_cont(c, ggml_permute(c, bias, 2, 0, 1, 3));                                  // [k, q, NH]
            T* mask = bias;
            if (sh) {
                T* shape = ggml_new_tensor_4d(c, GGML_TYPE_F32, ws2, ws2, NH, si.nW[1]);
                mask = ggml_add(c, ggml_repeat(c, bias, shape), shift);
            }
            tok = swin_block(c, m, p, tok, C, NH, si.nW[sh], gather[sh], keep[sh], scatter[sh], mask);
        }
        T* xo = ln(c, tok, m.get("bb.norm" + std::to_string(s) + ".weight"),
                   m.get("bb.norm" + std::to_string(s) + ".bias"));
        outs[s] = ggml_cont(c, ggml_transpose(c, xo));                                     // [N, C] == torch [C,H,W]
        out.C[s] = C; out.H[s] = H; out.W[s] = W;
        if (s < 3) {   // PatchMerging on the pre-norm block output
            const int Nh = (H / 2) * (W / 2);
            T* x[4];
            for (int q = 0; q < 4; ++q)
                x[q] = ggml_get_rows(c, tok, input(ggml_new_tensor_1d(c, GGML_TYPE_I32, Nh), si.merge[q].data()));
            const std::string dp = "bb.layers." + std::to_string(s) + ".downsample.";
            T* cat = ggml_concat(c, ggml_concat(c, x[0], x[1], 0), ggml_concat(c, x[2], x[3], 0), 0);   // [4C, Nh]
            cat = ln(c, cat, m.get(dp + "norm.weight"), m.get(dp + "norm.bias"));
            tok = ggml_mul_mat(c, m.get(dp + "reduction.weight"), cat);                                 // [2C, Nh]
            C *= 2; H /= 2; W /= 2;
        }
    }
    std::vector<std::vector<float>> r = run_graph(m, c, {outs[0], outs[1], outs[2], outs[3]}, ins);
    ggml_free(c);
    for (int s = 0; s < 4; ++s) out.f[s] = std::move(r[s]);
    return out;
}

// ============================ squeeze + decoder ============================
// Host feature map in torch [C,H,W] order (== ggml [W,H,C] same bytes).
struct Feat { std::vector<float> d; int C = 0, H = 0, W = 0; };

static Feat interp(const Feat& x, int Ho, int Wo) {                 // bilinear, align_corners=True
    Feat o; o.C = x.C; o.H = Ho; o.W = Wo; o.d.resize((size_t)x.C * Ho * Wo);
    double sy = Ho > 1 ? (double)(x.H - 1) / (Ho - 1) : 0.0, sx = Wo > 1 ? (double)(x.W - 1) / (Wo - 1) : 0.0;
    for (int ho = 0; ho < Ho; ++ho) { double fy = ho * sy; int y0 = (int)fy; int y1 = std::min(y0+1, x.H-1); double ly = fy - y0;
        for (int wo = 0; wo < Wo; ++wo) { double fx = wo * sx; int x0 = (int)fx; int x1 = std::min(x0+1, x.W-1); double lx = fx - x0;
            for (int ch = 0; ch < x.C; ++ch) { const float* s = &x.d[(size_t)ch*x.H*x.W];
                double v = (1-ly)*(1-lx)*s[(size_t)y0*x.W+x0] + (1-ly)*lx*s[(size_t)y0*x.W+x1]
                         + ly*(1-lx)*s[(size_t)y1*x.W+x0] + ly*lx*s[(size_t)y1*x.W+x1];
                o.d[(size_t)ch*Ho*Wo + (size_t)ho*Wo + wo] = (float)v; } } }
    return o;
}
static Feat image2patches(const Feat& img, int Href, int Wref) {  // 'b c (hg h)(wg w)->b (c hg wg) h w'
    int hg = img.H / Href, wg = img.W / Wref;
    Feat o; o.C = img.C * hg * wg; o.H = Href; o.W = Wref; o.d.resize((size_t)o.C * Href * Wref);
    for (int c = 0; c < img.C; ++c) for (int ih = 0; ih < hg; ++ih) for (int iw = 0; iw < wg; ++iw) {
        int oc = c*hg*wg + ih*wg + iw;
        for (int y = 0; y < Href; ++y) for (int x = 0; x < Wref; ++x)
            o.d[(size_t)oc*Href*Wref + (size_t)y*Wref + x] =
                img.d[(size_t)c*img.H*img.W + (size_t)(ih*Href+y)*img.W + (iw*Wref+x)];
    }
    return o;
}

// A decoder graph under construction: its context, the host buffers it uploads, and the pieces of
// arithmetic the decoder is made of. Every feature map is ggml [W,H,C], which is torch [C,H,W].
struct DecGraph {
    const Model& m;
    ggml_context* c = mkctx(4096);
    std::vector<std::pair<T*, const void*>> ins;

    explicit DecGraph(const Model& model) : m(model) {}
    ~DecGraph() { ggml_free(c); }
    DecGraph(const DecGraph&) = delete;
    DecGraph& operator=(const DecGraph&) = delete;

    T* in(const Feat& f) {
        T* t = ggml_new_tensor_3d(c, GGML_TYPE_F32, f.W, f.H, f.C);
        ggml_set_input(t); ins.push_back({t, f.d.data()});
        return t;
    }
    T* conv(const std::string& p, T* x, int pad) {                  // stride 1, bias if the model has one
        T* y = ggml_conv_2d(c, m.get(p + ".weight"), x, 1, 1, pad, pad, 1, 1);
        if (m.has(p + ".bias")) y = ggml_add(c, y, ggml_reshape_3d(c, m.get(p + ".bias"), 1, 1, y->ne[2]));
        return y;
    }
    T* bn(const std::string& p, T* x) {                             // folded BN: per-channel scale/shift
        const int64_t C = x->ne[2];
        return ggml_add(c, ggml_mul(c, x, ggml_reshape_3d(c, m.get(p + ".scale"), 1, 1, C)),
                        ggml_reshape_3d(c, m.get(p + ".shift"), 1, 1, C));
    }
    T* interp(T* x, int H, int W) {                                 // bilinear, align_corners=True
        return ggml_interpolate(c, x, W, H, x->ne[2], 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    }
    T* cat(std::vector<T*> xs) {                                    // along channels
        T* r = xs[0];
        for (size_t i = 1; i < xs.size(); ++i) r = ggml_concat(c, r, xs[i], 2);
        return r;
    }
    std::vector<Feat> run(std::vector<T*> outs) {
        std::vector<std::vector<float>> r = run_graph(m, c, outs, ins, 4096);
        std::vector<Feat> f(outs.size());
        for (size_t i = 0; i < outs.size(); ++i) {
            f[i].d = std::move(r[i]);
            f[i].W = (int)outs[i]->ne[0]; f[i].H = (int)outs[i]->ne[1]; f[i].C = (int)outs[i]->ne[2];
        }
        return f;
    }
};

// The four deformable branches of an ASPPDeformable, in the order the aspp concatenates them.
static const char* DEFORM_BRANCHES[4] = { ".aspp1", ".aspp_deforms.0", ".aspp_deforms.1", ".aspp_deforms.2" };

// One BasicDecBlk (conv_in -> ASPPDeformable -> conv_out), plus the GDT attention gate that follows
// decoder blocks 2-4. It runs as two graphs around the deformable convolutions, the one step ggml
// has no op for: the first builds the block's input and everything the deformable convs read, the
// second everything after them. `input` builds the block's input inside the first graph.
static Feat dec_block(const Model& m, const std::string& p, const std::function<T*(DecGraph&)>& input,
                      int gdt_level, int gpu) {
    const std::string a = p + ".dec_att";
    Feat o, offs[4], mods[4], pooled;
    {
        DecGraph g(m);
        T* x = ggml_relu(g.c, g.bn(p + ".bn_in", g.conv(p + ".conv_in", input(g), 1)));
        std::vector<T*> outs = { x };
        for (const char* branch : DEFORM_BRANCHES) {
            const std::string ac = a + branch + ".atrous_conv";
            const int K = (int)m.get(ac + ".regular_conv.weight")->ne[0];
            outs.push_back(g.conv(ac + ".offset_conv", x, K / 2));                          // [W,H,2K^2]
            outs.push_back(ggml_scale(g.c, ggml_sigmoid(g.c, g.conv(ac + ".modulator_conv", x, K / 2)), 2.0f));
        }
        // global_avg_pool: mean over the map -> 1x1 conv -> BN -> ReLU, one value per channel
        const std::string gp = a + ".global_avg_pool";
        T* w = m.get(gp + ".1.weight");                                                     // [1,1,Cin,OC]
        T* mean = ggml_mean(g.c, ggml_reshape_2d(g.c, ggml_cont(g.c, x), x->ne[0] * x->ne[1], x->ne[2]));  // [1, Cin]
        T* v = ggml_mul_mat(g.c, ggml_reshape_2d(g.c, w, w->ne[2], w->ne[3]), ggml_reshape_2d(g.c, mean, x->ne[2], 1));
        outs.push_back(ggml_relu(g.c, g.bn(gp + ".2", ggml_reshape_3d(g.c, v, 1, 1, w->ne[3]))));
        std::vector<Feat> r = g.run(outs);
        o = std::move(r[0]);
        for (int i = 0; i < 4; ++i) { offs[i] = std::move(r[1 + 2*i]); mods[i] = std::move(r[2 + 2*i]); }
        pooled = std::move(r[9]);
    }
    Feat d[4];
    for (int i = 0; i < 4; ++i) {
        T* w = m.get(a + DEFORM_BRANCHES[i] + ".atrous_conv.regular_conv.weight");
        const int K = (int)w->ne[0], OC = (int)w->ne[3];
        std::vector<float> wt = tensor_to_f32(w);                    // [KW,KH,Cin,OC] == [OC,Cin,K,K] C-order
        d[i].C = OC; d[i].H = o.H; d[i].W = o.W; d[i].d.resize((size_t)OC * o.H * o.W);
        deform_conv2d_run(o.d.data(), o.C, o.H, o.W, offs[i].d.data(), mods[i].d.data(), wt.data(), nullptr,
                          OC, K, d[i].d.data(), gpu);
    }
    DecGraph g(m);
    std::vector<T*> branches;
    for (int i = 0; i < 4; ++i) branches.push_back(ggml_relu(g.c, g.bn(a + DEFORM_BRANCHES[i] + ".bn", g.in(d[i]))));
    T* pool = g.in(pooled);                                                               // [1,1,OC]
    branches.push_back(ggml_repeat(g.c, pool, ggml_new_tensor_3d(g.c, GGML_TYPE_F32, o.W, o.H, pool->ne[2])));
    T* y = ggml_relu(g.c, g.bn(a + ".bn1", g.conv(a + ".conv1", g.cat(branches), 0)));
    y = g.bn(p + ".bn_out", g.conv(p + ".conv_out", y, 1));
    if (gdt_level > 0) {
        const std::string gl = "decoder.gdt_convs_" + std::to_string(gdt_level);
        T* h = ggml_relu(g.c, g.bn(gl + ".1", g.conv(gl + ".0", y, 1)));
        T* at = ggml_sigmoid(g.c, g.conv("decoder.gdt_convs_attn_" + std::to_string(gdt_level) + ".0", h, 0));
        y = ggml_mul(g.c, y, at);
    }
    return std::move(g.run({ y })[0]);
}

// simple_convs over the image cut into patches at the block's resolution (ipt_blk*).
static T* image_features(DecGraph& g, const std::string& blk, const Feat& patches) {
    const std::string p = "decoder." + blk;
    return g.conv(p + ".conv_out", g.conv(p + ".conv1", g.in(patches), 1), 1);             // two 3x3, no act
}

// The decoder's last stage as one graph: upsample decoder_block1's output to the image, concat the
// ipt_blk1 image features, and project with conv_out1 (1x1). The 1x1 projection, the concat and
// the bilinear upsample are all linear, so the projection is split per concat half and the
// upsample moved after it: interp(W*p) == W*interp(p). That upsamples one channel instead of 192,
// and the two 3x3 image convs run as direct convolutions instead of im2col stripes at 1024^2.
static std::vector<float> decoder_tail(const Model& m, const Feat& p, const Feat& img) {
    T* w = m.get("decoder.conv_out1.0.weight");                     // [1,1,IC,1]
    const int ICp = p.C, ICe = (int)w->ne[2] - p.C;
    ggml_context* c = mkctx(256);
    T* gp = ggml_new_tensor_3d(c, GGML_TYPE_F32, p.W, p.H, p.C);     ggml_set_input(gp);
    T* gi = ggml_new_tensor_3d(c, GGML_TYPE_F32, img.W, img.H, img.C); ggml_set_input(gi);
    auto conv = [&](T* k, T* x, int pad) { return ggml_conv_2d_direct(c, k, x, 1, 1, pad, pad, 1, 1); };
    auto bias = [&](T* y, const std::string& name) {
        return ggml_add(c, y, ggml_reshape_3d(c, m.get(name), 1, 1, m.get(name)->ne[0]));
    };
    T* wp = ggml_view_4d(c, w, 1, 1, ICp, 1, w->nb[1], w->nb[2], w->nb[3], 0);
    T* we = ggml_view_4d(c, w, 1, 1, ICe, 1, w->nb[1], w->nb[2], w->nb[3], (size_t)ICp * w->nb[2]);
    T* up = ggml_interpolate(c, conv(wp, gp, 0), img.W, img.H, 1, 1,
                             GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    T* e = bias(conv(m.get("decoder.ipt_blk1.conv1.weight"), gi, 1), "decoder.ipt_blk1.conv1.bias");
    e = bias(conv(m.get("decoder.ipt_blk1.conv_out.weight"), e, 1), "decoder.ipt_blk1.conv_out.bias");
    T* out = bias(ggml_add(c, up, conv(we, e, 0)), "decoder.conv_out1.0.bias");
    std::vector<float> r = run_graph(m, c, {out}, { {gp, p.d.data()}, {gi, img.d.data()} }, 256)[0];
    ggml_free(c);
    return r;
}

static Feat from_bb(const BBOut& b, int i) { Feat f; f.C=b.C[i]; f.H=b.H[i]; f.W=b.W[i]; f.d=b.f[i]; return f; }

std::vector<float> birefnet_matte(const Model& m, const std::vector<float>& chw1024, int gpu) {
    Feat img; img.C = 3; img.H = 1024; img.W = 1024; img.d = chw1024;
    // ---- backbone twice (mul_scl_ipt='cat') ----
    const BBOut full = swin_backbone(m, chw1024, 1024);
    const Feat img512 = interp(img, 512, 512);
    const BBOut half = swin_backbone(m, img512.d, 512);
    // xs[i] = cat(full_i, half_i upsampled to full_i's size): doubled channels
    std::vector<Feat> xs;
    {
        Feat f[4], h[4];
        for (int i = 0; i < 4; ++i) { f[i] = from_bb(full, i); h[i] = from_bb(half, i); }
        DecGraph g(m);
        std::vector<T*> outs;
        for (int i = 0; i < 4; ++i) outs.push_back(g.cat({ g.in(f[i]), g.interp(g.in(h[i]), f[i].H, f[i].W) }));
        xs = g.run(outs);
    }
    const Feat pat32 = image2patches(img, 32, 32), pat64 = image2patches(img, 64, 64),
               pat128 = image2patches(img, 128, 128), pat256 = image2patches(img, 256, 256);

    // ---- cxt cat on x4 + squeeze ----
    const Feat x4 = dec_block(m, "squeeze_module.0", [&](DecGraph& g) {
        const int H = xs[3].H, W = xs[3].W;
        return g.cat({ g.interp(g.in(xs[0]), H, W), g.interp(g.in(xs[1]), H, W), g.interp(g.in(xs[2]), H, W),
                       g.in(xs[3]) });                                                   // 5760@32
    }, 0, gpu);                                                                          // 3072@32

    // ---- decoder ----
    const Feat p4 = dec_block(m, "decoder.decoder_block4", [&](DecGraph& g) {
        return g.cat({ g.in(x4), image_features(g, "ipt_blk5", pat32) });               // 3456@32
    }, 4, gpu);                                                                          // 1536@32
    // Each later block: the previous output upsampled plus a lateral 1x1 of the backbone feature,
    // then the image features at that resolution.
    auto next = [](const Feat& prev, const Feat& lateral_in, const std::string& lateral, const std::string& ipt,
                   const Feat& patches) {
        return [prev = &prev, lat = &lateral_in, patches = &patches, lateral, ipt](DecGraph& g) {
            T* up = ggml_add(g.c, g.interp(g.in(*prev), lat->H, lat->W),
                             g.conv("decoder." + lateral + ".conv", g.in(*lat), 0));
            return g.cat({ up, image_features(g, ipt, *patches) });
        };
    };
    const Feat p3 = dec_block(m, "decoder.decoder_block3", next(p4, xs[2], "lateral_block4", "ipt_blk4", pat64), 3, gpu);
    const Feat p2 = dec_block(m, "decoder.decoder_block2", next(p3, xs[1], "lateral_block3", "ipt_blk3", pat128), 2, gpu);
    const Feat p1 = dec_block(m, "decoder.decoder_block1", next(p2, xs[0], "lateral_block2", "ipt_blk2", pat256), 0, gpu);
    return decoder_tail(m, p1, img);                                                     // 1@1024
}

} // namespace trellis
