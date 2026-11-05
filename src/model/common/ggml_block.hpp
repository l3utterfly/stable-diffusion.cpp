#ifndef __SD_MODEL_COMMON_GGML_BLOCK_HPP__
#define __SD_MODEL_COMMON_GGML_BLOCK_HPP__

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "model.h"

// ggml_qknorm_rope asserts on these, so callers check before building the node.
__STATIC_INLINE__ bool qknorm_rope_inputs_ok(ggml_tensor* x, ggml_tensor* weight, ggml_tensor* theta) {
    return x != nullptr && weight != nullptr && theta != nullptr &&
           x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && theta->type == GGML_TYPE_F32 &&
           ggml_nelements(weight) == x->ne[0] && ggml_nelements(theta) >= x->ne[2] * 2 * x->ne[0];
}

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2TensorStorage& tensor_storage_map, ggml_type default_type) {
        ggml_type wtype = default_type;
        auto iter       = tensor_storage_map.find(name);
        if (iter != tensor_storage_map.end()) {
            const TensorStorage& tensor_storage = iter->second;
            if (tensor_storage.expected_type != GGML_TYPE_COUNT) {
                wtype = tensor_storage.expected_type;
            } else {
                wtype = tensor_storage.type;
            }
        }
        return wtype;
    }

    void init_blocks(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_storage_map, prefix + pair.first);
        }
    }

    virtual void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {}

    virtual enum ggml_op param_usage_op(const std::string& name) const {
        (void)name;
        return GGML_OP_NONE;
    }

public:
    void init(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_params(ctx, tensor_storage_map, prefix);
        init_blocks(ctx, tensor_storage_map, prefix);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            ggml_tensor* param           = pair.second;
            tensors[prefix + pair.first] = pair.second;
            ggml_set_name(param, (prefix + pair.first).c_str());
        }
    }

    void get_param_tensor_ops(std::map<ggml_tensor*, enum ggml_op>& tensor_ops) {
        for (auto& pair : blocks) {
            pair.second->get_param_tensor_ops(tensor_ops);
        }
        for (auto& pair : params) {
            enum ggml_op op = param_usage_op(pair.first);
            if (op != GGML_OP_NONE) {
                tensor_ops[pair.second] = op;
            }
        }
    }

    virtual std::string get_desc() {
        return "GGMLBlock";
    }

    void get_all_blocks(std::vector<GGMLBlock*>& result) {
        result.push_back(this);
        for (auto& block_iter : blocks) {
            if (block_iter.second) {
                block_iter.second->get_all_blocks(result);
            }
        }
    }
};

class UnaryBlock : public GGMLBlock {
public:
    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) = 0;
};

class Identity : public UnaryBlock {
public:
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        return x;
    }
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;
    bool force_prec_f32;
    bool has_weight_scale       = false;
    bool int8_convrot           = false;
    int int8_convrot_group_size = 0;
    float scale;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix            = prefix;
        has_weight_scale        = false;
        int8_convrot            = false;
        int8_convrot_group_size = 0;
        enum ggml_type wtype    = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
        auto weight_storage           = tensor_storage_map.find(prefix + "weight");
        const bool is_int8_tensorwise = weight_storage != tensor_storage_map.end() && weight_storage->second.is_int8_tensorwise;
        auto weight_scale_storage     = tensor_storage_map.find(prefix + "weight_scale");
        if (weight_scale_storage != tensor_storage_map.end()) {
            const int64_t scale_nelements = weight_scale_storage->second.nelements();
            GGML_ASSERT(scale_nelements == 1 || scale_nelements == out_features);
            params["weight_scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, scale_nelements);
            has_weight_scale       = true;
        }
        if (is_int8_tensorwise) {
            GGML_ASSERT(wtype == GGML_TYPE_I8);
            GGML_ASSERT(has_weight_scale);
            int8_convrot            = weight_storage->second.int8_convrot;
            int8_convrot_group_size = weight_storage->second.int8_convrot_group_size;
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias           = true,
           bool force_f32      = false,
           bool force_prec_f32 = false,
           float scale         = 1.f)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32),
          scale(scale) {}

    void set_scale(float scale_) {
        scale = scale_;
    }

    void set_force_prec_f32(bool force_prec_f32_) {
        force_prec_f32 = force_prec_f32_;
    }

    ggml_type weight_type() const {
        auto it = params.find("weight");
        return it != params.end() && it->second ? it->second->type : GGML_TYPE_F32;
    }

    bool has_quantized_weight() const {
        return ggml_is_quantized(weight_type());
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w            = params["weight"];
        const float scale         = ctx->linear_scale > 0.f ? ctx->linear_scale : this->scale;
        ggml_tensor* weight_scale = has_weight_scale ? params["weight_scale"] : nullptr;
#ifndef SD_USE_UPSTREAM_GGML
        if (w->type == GGML_TYPE_F8_E4M3 || w->type == GGML_TYPE_F8_E5M2) {
            bool supports_fp8_matmul = false;
            if (ctx->backend != nullptr) {
                ggml_tensor* fp8_matmul = ggml_mul_mat(ctx->ggml_ctx, w, x);
                if (force_prec_f32) {
                    ggml_prec_set_acc(fp8_matmul, GGML_PREC_F32);
                }
                supports_fp8_matmul = ggml_backend_supports_op(ctx->backend, fp8_matmul) && !sd_backend_is_cpu(ctx->backend);
            }
            if (!supports_fp8_matmul) {
                w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_BF16);
            }
        }
#endif
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        ggml_tensor* linear_bias = has_weight_scale ? nullptr : b;
        ggml_tensor* out         = nullptr;
        if (w->type == GGML_TYPE_I8) {
            if (x->type != GGML_TYPE_F32) {
                x = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, x);
            }
            if (!ggml_is_contiguous(x)) {
                x = ggml_cont(ctx->ggml_ctx, x);
            }
            ggml_tensor* lora_input = x;
            if (ctx->weight_adapter && b != nullptr) {
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
#ifndef SD_USE_UPSTREAM_GGML
            if (int8_convrot && scale == 1.f) {
                const auto cache_key = std::make_pair(x, int8_convrot_group_size);
                auto cached          = ctx->int8_convrot_cache.find(cache_key);
                if (cached == ctx->int8_convrot_cache.end()) {
                    x = ggml_quantize_i8_convrot(ctx->ggml_ctx, x, int8_convrot_group_size);
                    ctx->int8_convrot_cache.emplace(cache_key, x);
                } else {
                    x = cached->second;
                }
            }
#endif
            out = ggml_ext_linear_i8_tensorwise(ctx->ggml_ctx,
                                                x,
                                                w,
                                                weight_scale,
                                                b,
                                                int8_convrot ? int8_convrot_group_size : 0,
                                                scale);
            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
                forward_params.linear.force_prec_f32 = force_prec_f32;
                forward_params.linear.scale          = scale;
                out                                  = ctx->weight_adapter->add_lora_to_output(ctx->ggml_ctx,
                                                                                               ctx->backend,
                                                                                               lora_input,
                                                                                               w,
                                                                                               out,
                                                                                               prefix,
                                                                                               forward_params);
            }
            return out;
        }
        if (has_weight_scale) {
            out = ggml_ext_linear(ctx->ggml_ctx, x, w, nullptr, force_prec_f32, scale);
            out = ggml_mul(ctx->ggml_ctx, out, weight_scale);
            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
                forward_params.linear.force_prec_f32 = force_prec_f32;
                forward_params.linear.scale          = scale;
                out                                  = ctx->weight_adapter->add_lora_to_output(ctx->ggml_ctx,
                                                                                               ctx->backend,
                                                                                               x,
                                                                                               w,
                                                                                               out,
                                                                                               prefix,
                                                                                               forward_params);
                if (b != nullptr) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
            if (b != nullptr) {
                out = ggml_add_inplace(ctx->ggml_ctx, out, b);
            }
            return out;
        }
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
            forward_params.linear.force_prec_f32 = force_prec_f32;
            forward_params.linear.scale          = scale;
            out                                  = ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, linear_bias, prefix, forward_params);
        } else {
            out = ggml_ext_linear(ctx->ggml_ctx, x, w, linear_bias, force_prec_f32, scale);
        }
        return out;
    }

    ggml_tensor* forward_segmented(GGMLRunnerContext* ctx,
                                   ggml_tensor* x0,
                                   ggml_tensor* x1) {
        ggml_tensor* w = params["weight"];
        const bool segmented_weight = w->type == GGML_TYPE_F8_E4M3 ||
                                      w->type == GGML_TYPE_Q4_0 ||
                                      w->type == GGML_TYPE_MXFP4;
        if (ctx->weight_adapter == nullptr && scale == 1.f &&
            segmented_weight) {
            ggml_tensor* out = ggml_mul_mat_segmented(ctx->ggml_ctx, w, x0, x1);
            if (force_prec_f32) {
                ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            }
            if (ctx->backend != nullptr && ggml_backend_supports_op(ctx->backend, out)) {
                if (has_weight_scale) {
                    out = ggml_mul(ctx->ggml_ctx, out, params["weight_scale"]);
                }
                if (bias) {
                    out = ggml_add_inplace(ctx->ggml_ctx, out, params["bias"]);
                }
                return out;
            }
        }

        return forward(ctx, ggml_concat(ctx->ggml_ctx, x0, x1, 0));
    }
};

__STATIC_INLINE__ bool support_get_rows(ggml_type wtype) {
    switch (wtype) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    bool is_int8_tensorwise     = false;
    int int8_convrot_group_size = 0;

    static void get_rows_i8(ggml_tensor* dst, int ith, int nth, void* userdata) {
        const auto* embedding     = static_cast<const Embedding*>(userdata);
        const int group_size      = embedding->int8_convrot_group_size;
        const auto* weight        = dst->src[0];
        const auto* input_ids     = static_cast<const int32_t*>(dst->src[1]->data);
        const auto* scales        = static_cast<const float*>(dst->src[2]->data);
        const bool scalar_scale   = ggml_nelements(dst->src[2]) == 1;
        const float normalization = group_size > 0 ? 1.f / std::sqrt(static_cast<float>(group_size)) : 1.f;
        for (int64_t row = ith; row < dst->ne[1]; row += nth) {
            const int32_t token = input_ids[row];
            GGML_ASSERT(token >= 0 && token < weight->ne[1]);
            const auto* src   = reinterpret_cast<const int8_t*>(static_cast<const char*>(weight->data) + token * weight->nb[1]);
            auto* out         = reinterpret_cast<float*>(static_cast<char*>(dst->data) + row * dst->nb[1]);
            const float scale = scales[scalar_scale ? 0 : token] * normalization;
            for (int64_t i = 0; i < weight->ne[0]; ++i) {
                out[i] = static_cast<float>(src[i]) * scale;
            }
            // The regular Hadamard rotation is symmetric and its own inverse.
            for (int stride = 1; stride < group_size; stride *= 4) {
                for (int64_t base = 0; base < weight->ne[0]; base += 4 * stride) {
                    for (int j = 0; j < stride; ++j) {
                        float* values      = out + base + j;
                        const float a      = values[0];
                        const float b      = values[stride];
                        const float c      = values[2 * stride];
                        const float d      = values[3 * stride];
                        values[0]          = a + b + c - d;
                        values[stride]     = a + b - c + d;
                        values[2 * stride] = a - b + c + d;
                        values[3 * stride] = -a + b + c + d;
                    }
                }
            }
        }
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        auto weight_storage     = tensor_storage_map.find(prefix + "weight");
        is_int8_tensorwise      = weight_storage != tensor_storage_map.end() && weight_storage->second.is_int8_tensorwise;
        int8_convrot_group_size = 0;
        enum ggml_type wtype    = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (is_int8_tensorwise) {
            GGML_ASSERT(wtype == GGML_TYPE_I8);
            auto scale_storage = tensor_storage_map.find(prefix + "weight_scale");
            GGML_ASSERT(scale_storage != tensor_storage_map.end());
            const int64_t scale_nelements = scale_storage->second.nelements();
            GGML_ASSERT(scale_nelements == 1 || scale_nelements == num_embeddings);
            params["weight_scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, scale_nelements);
            if (weight_storage->second.int8_convrot) {
                int8_convrot_group_size = weight_storage->second.int8_convrot_group_size;
                int remainder           = int8_convrot_group_size;
                while (remainder > 1 && remainder % 4 == 0) {
                    remainder /= 4;
                }
                GGML_ASSERT(remainder == 1 && embedding_dim % int8_convrot_group_size == 0);
            }
        } else if (!support_get_rows(wtype)) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

    enum ggml_op param_usage_op(const std::string& name) const override {
        if (is_int8_tensorwise) {
            return GGML_OP_CUSTOM;
        }
        return name == "weight" ? GGML_OP_GET_ROWS : GGML_OP_NONE;
    }

public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : embedding_dim(embedding_dim),
          num_embeddings(num_embeddings) {
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids) override {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        ggml_tensor* embedding;
        if (is_int8_tensorwise) {
            GGML_ASSERT(input_ids->type == GGML_TYPE_I32);
            ggml_tensor* args[] = {weight, input_ids, params["weight_scale"]};
            embedding           = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F32, embedding_dim, input_ids->ne[0], 1, 1,
                                                 args, 3, get_rows_i8, GGML_N_TASKS_MAX, this);
        } else {
            input_ids = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
            embedding = ggml_get_rows(ctx->ggml_ctx, weight, input_ids);
        }
        embedding = ggml_reshape_3d(ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

        // [N, n_token, embedding_dim]
        return embedding;
    }
};

class Conv1d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int64_t groups;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool bias;
    bool force_prec_f32;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        ggml_type wtype  = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16);
        params["weight"] = ggml_new_tensor_3d(ctx, wtype, kernel_size, in_channels / groups, out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    Conv1d(int64_t in_channels,
           int64_t out_channels,
           int kernel_size,
           int stride          = 1,
           int padding         = 0,
           int dilation        = 1,
           int64_t groups      = 1,
           bool bias           = true,
           bool force_prec_f32 = false)
        : in_channels(in_channels),
          out_channels(out_channels),
          groups(groups),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias),
          force_prec_f32(force_prec_f32) {
        GGML_ASSERT(in_channels > 0 && out_channels > 0 && groups > 0);
        GGML_ASSERT(in_channels % groups == 0 && out_channels % groups == 0);
        GGML_ASSERT(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0);
    }

    std::string get_desc() override {
        return "Conv1d";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        GGML_ASSERT(x->ne[1] == in_channels);
        return ggml_ext_conv_1d(ctx->ggml_ctx, x, params["weight"], bias ? params["bias"] : nullptr,
                                stride, padding, dilation, groups, force_prec_f32);
    }
};

class Conv2d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride   = {1, 1},
           std::pair<int, int> padding  = {0, 0},
           std::pair<int, int> dilation = {1, 1},
           bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() override {
        return "Conv2d";
    }

    // The direct HMX kernel reads tiled weights and supports stride 1. Keep every
    // second sample of its pad-1 result. UNet samples even coordinates, while
    // VAE downsamples an input with one pixel of zero padding at the right/bottom.
    ggml_tensor* try_forward_downsample2(GGMLRunnerContext* ctx, ggml_tensor* x, bool vae) {
        const int pad = vae ? 0 : 1;
        if (!bias || has_weight_adapter(ctx) || ctx->circular_x_enabled || ctx->circular_y_enabled ||
            (!std::isfinite(scale) || scale == 0.f) || kernel_size != std::make_pair(3, 3) || stride != std::make_pair(2, 2) ||
            padding != std::make_pair(pad, pad) || dilation != std::make_pair(1, 1) ||
            (vae && (!ctx->conv2d_direct_enabled || x->ne[0] < 2 || x->ne[1] < 2)) ||
            !sd_backend_is(ctx->backend, "HTP")) {
            return nullptr;
        }
        auto c = ctx->ggml_ctx;
        auto full = scale == 1.f
                        ? ggml_conv_2d_direct_bias(c, params["weight"], x, params["bias"], 1, 1, 1, 1, 1, 1)
                        : ggml_conv_2d_direct(c, params["weight"], ggml_ext_scale(c, x, scale), 1, 1, 1, 1, 1, 1);
        if (!ggml_backend_supports_op(ctx->backend, full)) {
            return nullptr;
        }
        const int64_t ow = (full->ne[0] + pad) / 2;
        const int64_t oh = (full->ne[1] + pad) / 2;
        // A singleton first axis permits striding in x without changing the plane stride.
        auto view = ggml_view_4d(c, full, 1, ow, oh, full->ne[2] * full->ne[3],
                                 2 * full->nb[0], 2 * full->nb[1], full->nb[2],
                                 vae ? full->nb[0] + full->nb[1] : 0);
        auto out = ggml_reshape_4d(c, ggml_cont(c, view), ow, oh, full->ne[2], full->ne[3]);
        if (scale != 1.f) {
            out = ggml_ext_scale(c, out, 1.f / scale);
            out = add_scaled_bias(ctx, out, params["bias"]);
        }
        return out;
    }

    ggml_tensor* forward_vae_downsample2(GGMLRunnerContext* ctx, ggml_tensor* x) {
        if (auto out = try_forward_downsample2(ctx, x, true)) {
            return out;
        }
        x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
        return forward(ctx, x);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        if (auto out = try_forward_downsample2(ctx, x, false)) {
            return out;
        }
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        if (has_weight_adapter(ctx)) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
            forward_params.conv2d.s0         = stride.second;
            forward_params.conv2d.s1         = stride.first;
            forward_params.conv2d.p0         = padding.second;
            forward_params.conv2d.p1         = padding.first;
            forward_params.conv2d.d0         = dilation.second;
            forward_params.conv2d.d1         = dilation.first;
            forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
            forward_params.conv2d.circular_x = ctx->circular_x_enabled;
            forward_params.conv2d.circular_y = ctx->circular_y_enabled;
            forward_params.conv2d.scale      = scale;
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        if (b != nullptr && ctx->backend != nullptr && b->type == GGML_TYPE_F32 && ggml_nelements(b) == w->ne[3] &&
            !ctx->circular_x_enabled && !ctx->circular_y_enabled && scale == 1.f) {
            ggml_tensor* out = ggml_conv_2d_direct_bias(
                ctx->ggml_ctx, w, x, b,
                stride.second, stride.first, padding.second, padding.first,
                dilation.second, dilation.first);
            if (ggml_backend_supports_op(ctx->backend, out)) {
                return out;
            }
        }
        if (sd_backend_is(ctx->backend, "HTP")) {
            // sd.cpp also uses bias-free direct convolution for scaled VAE layers.
            // Restore the input scale before adding the bias.
            if (ctx->conv2d_direct_enabled && !ctx->circular_x_enabled && !ctx->circular_y_enabled && scale != 0.f) {
                auto c      = ctx->ggml_ctx;
                auto scaled = scale == 1.f ? x : ggml_ext_scale(c, x, scale);
                auto out    = ggml_conv_2d_direct(c, w, scaled, stride.second, stride.first, padding.second,
                                                 padding.first, dilation.second, dilation.first);
                if (ggml_backend_supports_op(ctx->backend, out)) {
                    if (scale != 1.f) {
                        out = ggml_ext_scale(c, out, 1.f / scale);
                    }
                    return b ? add_scaled_bias(ctx, out, b) : out;
                }
            }
            if ((kernel_size == std::make_pair(1, 1) || kernel_size == std::make_pair(3, 3)) && in_channels % 32 == 0) {
                static std::once_flag warned;
                std::call_once(warned, [&] {
                    LOG_ERROR("%s: conv (k=%dx%d s=%dx%d p=%dx%d) falls back from the HMX direct kernel "
                              "but its weight is HMX-repacked; the result will be wrong",
                              prefix.c_str(), kernel_size.first, kernel_size.second,
                              stride.first, stride.second, padding.first, padding.second);
                });
            }
        }
        // Supported HTP direct convolutions returned above. A rejected shape
        // (e.g. the VAE's 16-channel input) must use im2col: CPU direct fallback
        // would leave the following in-place bias add aliasing a CPU buffer.
        const bool fallback_direct = ctx->conv2d_direct_enabled && !sd_backend_is(ctx->backend, "HTP");
        return ggml_ext_conv_2d(ctx->ggml_ctx,
                                x,
                                w,
                                b,
                                stride.second,
                                stride.first,
                                padding.second,
                                padding.first,
                                dilation.second,
                                dilation.first,
                                fallback_direct,
                                ctx->circular_x_enabled,
                                ctx->circular_y_enabled,
                                scale);
    }

    ggml_tensor* forward_upscale(GGMLRunnerContext* ctx,
                                 ggml_tensor* x,
                                 int upscale_factor) {
        ggml_tensor* out = try_forward_upscale(ctx, x, upscale_factor);
        if (out == nullptr) {
            return forward(ctx, ggml_upscale(ctx->ggml_ctx, x, upscale_factor,
                                             GGML_SCALE_MODE_NEAREST));
        }

        return out;
    }

    bool supports_upscale(GGMLRunnerContext* ctx,
                          ggml_tensor* x,
                          int upscale_factor) {
        return try_forward_upscale(ctx, x, upscale_factor) != nullptr;
    }

    ggml_tensor* try_forward_upscale_add(GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* residual,
                                         int residual_factor_t,
                                         int upscale_factor) {
        return try_forward_upscale(ctx, x, upscale_factor,
                                   residual, residual_factor_t);
    }

private:
    bool has_weight_adapter(const GGMLRunnerContext* ctx) const {
        return ctx->weight_adapter &&
               (ctx->weight_adapter->affects_tensor(prefix + "weight") ||
                (bias && ctx->weight_adapter->affects_tensor(prefix + "bias")));
    }

    ggml_tensor* try_forward_upscale(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     int upscale_factor,
                                     ggml_tensor* residual = nullptr,
                                     int residual_factor_t = 0) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = bias ? params["bias"] : nullptr;
        if (ctx->backend == nullptr ||
            has_weight_adapter(ctx) || ctx->circular_x_enabled ||
            ctx->circular_y_enabled || (scale != 1.f && (!std::isfinite(scale) || scale == 0.f || !sd_backend_is(ctx->backend, "HTP"))) ||
            upscale_factor <= 1 || w->ne[2] != x->ne[2] ||
            (b != nullptr && (b->type != GGML_TYPE_F32 || ggml_nelements(b) != w->ne[3])) ||
            (residual != nullptr && residual_factor_t != 1 && residual_factor_t != 2)) {
            return nullptr;
        }

        // The F16 fused kernel rounds its bias before adding. Scaled F16 VAE
        // layers need the existing mixed F16+F32 add after restoring the scale.
        auto deferred_bias = scale != 1.f && x->type == GGML_TYPE_F16 && residual == nullptr ? b : nullptr;
        if (deferred_bias != nullptr) {
            b = nullptr;
        }
        if (scale != 1.f) {
            x = ggml_ext_scale(ctx->ggml_ctx, x, scale);
            // Bias and the compact residual join the fused sum before its scale is restored.
            if (b != nullptr) {
                b = ggml_ext_scale(ctx->ggml_ctx, b, scale);
            }
            if (residual != nullptr) {
                residual = ggml_ext_scale(ctx->ggml_ctx, residual, scale);
            }
        }
        ggml_tensor* out = residual == nullptr
                               ? ggml_conv_2d_direct_upscale(
                                     ctx->ggml_ctx, w, x, b, upscale_factor,
                                     stride.second, stride.first, padding.second, padding.first,
                                     dilation.second, dilation.first)
                               : ggml_conv_2d_direct_upscale_add(
                                     ctx->ggml_ctx, w, x, b,
                                     residual, residual_factor_t, upscale_factor,
                                     stride.second, stride.first, padding.second, padding.first,
                                     dilation.second, dilation.first);
        if (!ggml_backend_supports_op(ctx->backend, out)) {
            return nullptr;
        }
        if (scale != 1.f) {
            out = ggml_ext_scale(ctx->ggml_ctx, out, 1.f / scale);
        }
        return deferred_bias ? add_scaled_bias(ctx, out, deferred_bias) : out;
    }

    ggml_tensor* add_scaled_bias(GGMLRunnerContext* ctx, ggml_tensor* out, ggml_tensor* b) {
        auto c = ctx->ggml_ctx;
        const bool widen = scale != 1.f && sd_backend_is(ctx->backend, "HTP") &&
                           out->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F32;
        // HTP binary ops require matching types. Preserve CPU mixed-add semantics:
        // promote the exact half value, add the original float bias, then round
        // once to half. Casting the bias first changes half-way rounding cases.
        if (widen) {
            out = ggml_cast(c, out, GGML_TYPE_F32);
        }
        out = ggml_add_inplace(c, out, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
        return widen ? ggml_cast(c, out, GGML_TYPE_F16) : out;
    }
};

class Conv2d_grouped : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int groups;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels / groups, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d_grouped(int64_t in_channels,
                   int64_t out_channels,
                   int groups,
                   std::pair<int, int> kernel_size,
                   std::pair<int, int> stride   = {1, 1},
                   std::pair<int, int> padding  = {0, 0},
                   std::pair<int, int> dilation = {1, 1},
                   bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          groups(groups),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() override {
        return "Conv2d_grouped";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }

        if (groups == 1) {
            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
            }
            return ggml_ext_conv_2d(ctx->ggml_ctx, x, w, b,
                                    stride.second, stride.first,
                                    padding.second, padding.first,
                                    dilation.second, dilation.first,
                                    ctx->conv2d_direct_enabled,
                                    ctx->circular_x_enabled,
                                    ctx->circular_y_enabled,
                                    scale);
        }

        if (groups == in_channels && groups == out_channels) {
            ggml_tensor* res;
            if (ctx->conv2d_direct_enabled) {
                res = ggml_conv_2d_dw_direct(ctx->ggml_ctx, w, x,
                                             stride.second, stride.first,
                                             padding.second, padding.first,
                                             dilation.second, dilation.first);
            } else {
                res = ggml_conv_2d_dw(ctx->ggml_ctx, w, x,
                                      stride.second, stride.first,
                                      padding.second, padding.first,
                                      dilation.second, dilation.first);
            }
            if (b) {
                b   = ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, b->ne[0], 1);
                res = ggml_add_inplace(ctx->ggml_ctx, res, b);
            }
            return res;
        }

        int64_t ic_g = in_channels / groups;
        int64_t oc_g = out_channels / groups;

        std::vector<ggml_tensor*> out_slices(groups);

        for (int i = 0; i < groups; ++i) {
            size_t x_offset  = i * ic_g * x->nb[2];
            ggml_tensor* x_i = ggml_view_4d(ctx->ggml_ctx, x,
                                            x->ne[0], x->ne[1], ic_g, x->ne[3],
                                            x->nb[1], x->nb[2], x->nb[3],
                                            x_offset);

            size_t w_offset  = i * oc_g * w->nb[3];
            ggml_tensor* w_i = ggml_view_4d(ctx->ggml_ctx, w,
                                            w->ne[0], w->ne[1], w->ne[2], oc_g,
                                            w->nb[1], w->nb[2], w->nb[3],
                                            w_offset);

            ggml_tensor* b_i = nullptr;
            if (b) {
                size_t b_offset = i * oc_g * b->nb[0];
                b_i             = ggml_view_1d(ctx->ggml_ctx, b, oc_g, b_offset);
            }

            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                out_slices[i]                    = ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x_i, w_i, b_i, prefix, forward_params);
            } else {
                out_slices[i] = ggml_ext_conv_2d(ctx->ggml_ctx, x_i, w_i, b_i,
                                                 stride.second, stride.first,
                                                 padding.second, padding.first,
                                                 dilation.second, dilation.first,
                                                 ctx->conv2d_direct_enabled,
                                                 ctx->circular_x_enabled,
                                                 ctx->circular_y_enabled,
                                                 scale);
            }
        }

        ggml_tensor* out = ggml_ext_vec_concat(ctx->ggml_ctx, out_slices, 2);

        return out;
    }
};

class Conv3d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::tuple<int, int, int> kernel_size;
    std::tuple<int, int, int> stride;
    std::tuple<int, int, int> padding;
    std::tuple<int, int, int> dilation;
    bool bias;
    bool force_prec_f32;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           std::tuple<int, int, int> kernel_size,
           std::tuple<int, int, int> stride   = {1, 1, 1},
           std::tuple<int, int, int> padding  = {0, 0, 0},
           std::tuple<int, int, int> dilation = {1, 1, 1},
           bool bias                          = true,
           bool force_prec_f32                = false)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias),
          force_prec_f32(force_prec_f32) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            if (w->type != GGML_TYPE_F16) {
                w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F16);
            }
        }
        if (bias) {
            b = params["bias"];
            if (ctx->weight_adapter) {
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_conv_3d(ctx->ggml_ctx, ctx->backend, x, w, b, in_channels,
                                std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                std::get<2>(padding), std::get<1>(padding), std::get<0>(padding),
                                std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                force_prec_f32, ctx->conv3d_direct_enabled);
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;
                params["bias"]       = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int64_t normalized_shape,
              float eps               = 1e-05f,
              bool elementwise_affine = true,
              bool bias               = true)
        : normalized_shape(normalized_shape),
          eps(eps),
          elementwise_affine(elementwise_affine),
          bias(bias) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;

        if (elementwise_affine) {
            w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            if (bias) {
                b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
        }
        return ggml_ext_layer_norm(ctx->ggml_ctx, x, w, b, eps);
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;
    std::string prefix;

    void get_affine_params(GGMLRunnerContext* ctx,
                           ggml_tensor** weight,
                           ggml_tensor** bias) {
        *weight = nullptr;
        *bias   = nullptr;
        if (!affine) {
            return;
        }

        *weight = params["weight"];
        *bias   = params["bias"];
        if (ctx->weight_adapter) {
            *weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, *weight, prefix + "weight");
            *bias   = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, *bias, prefix + "bias");
        }
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x,
                         bool inplace = false) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;
        get_affine_params(ctx, &w, &b);
        return ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups, eps, inplace);
    }

    ggml_tensor* forward_silu(GGMLRunnerContext* ctx, ggml_tensor* x,
                              bool inplace = false) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;
        get_affine_params(ctx, &w, &b);
        if (ctx->backend != nullptr && w != nullptr && b != nullptr &&
            w->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
            ggml_nelements(w) == x->ne[2] && ggml_nelements(b) == x->ne[2]) {
            ggml_tensor* out = inplace
                                   ? ggml_group_norm_affine_silu_inplace(ctx->ggml_ctx, x, w, b, num_groups, eps)
                                   : ggml_group_norm_affine_silu(ctx->ggml_ctx, x, w, b, num_groups, eps);
            if (ggml_backend_supports_op(ctx->backend, out)) {
                return out;
            }
        }

        x = ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups, eps, inplace);
        return ggml_silu_inplace(ctx->ggml_ctx, x);
    }
};

class GroupNorm32 : public GroupNorm {
public:
    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    bool elementwise_affine;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        this->prefix = prefix;
        if (!elementwise_affine) {
            return;
        }
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps               = 1e-06f,
            bool elementwise_affine = true)
        : hidden_size(hidden_size),
          eps(eps),
          elementwise_affine(elementwise_affine) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        if (!elementwise_affine) {
            return ggml_rms_norm(ctx->ggml_ctx, x, eps);
        }
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
        }
        x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        x = ggml_mul_inplace(ctx->ggml_ctx, x, w);
        return x;
    }

    ggml_tensor* try_forward_rope(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* theta, bool interleaved = true) {
        if (ctx->backend == nullptr) {
            return nullptr;
        }
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
        }
        if (!qknorm_rope_inputs_ok(x, w, theta)) {
            return nullptr;
        }
        if (!interleaved) {
            // The fused kernel rotates adjacent pairs. For split-half RoPE, apply
            // the same channel permutation to Q, K and their affine weights;
            // attention scores are invariant under this shared permutation.
            if (x->ne[0] % 2 != 0 || x->ne[3] != 1 || x->nb[0] != ggml_element_size(x)) {
                return nullptr;
            }
            auto c           = ctx->ggml_ctx;
            const auto d     = x->ne[0];
            const auto heads = x->ne[1];
            const auto len   = x->ne[2];
            x = ggml_view_4d(c, x, d / 2, 2, heads, len, d / 2 * x->nb[0], x->nb[1], x->nb[2], 0);
            x = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3)), d, heads, len, 1);
            w = ggml_reshape_1d(c, ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, w, d / 2, 2))), d);
        }
        ggml_tensor* out = ggml_qknorm_rope(ctx->ggml_ctx, x, w, theta, eps);
        return ggml_backend_supports_op(ctx->backend, out) ? out : nullptr;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    bool proj_in;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string in_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t n_head,
                       bool qkv_proj_bias        = true,
                       bool out_proj_bias        = true,
                       bool proj_in              = false,
                       std::string q_proj_name   = "q_proj",
                       std::string k_proj_name   = "k_proj",
                       std::string v_proj_name   = "v_proj",
                       std::string in_proj_name  = "in_proj",
                       std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim),
          n_head(n_head),
          proj_in(proj_in),
          q_proj_name(q_proj_name),
          k_proj_name(k_proj_name),
          v_proj_name(v_proj_name),
          in_proj_name(in_proj_name),
          out_proj_name(out_proj_name) {
        if (proj_in) {
            blocks[in_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim * 3, qkv_proj_bias));
        } else {
            blocks[q_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[k_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[v_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        }
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* mask = nullptr) {
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (proj_in) {
            auto in_proj = std::dynamic_pointer_cast<Linear>(blocks[in_proj_name]);
            auto qkv     = in_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            q            = qkv_vec[0];
            k            = qkv_vec[1];
            v            = qkv_vec[2];
        } else {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);

            q = q_proj->forward(ctx, x);
            k = k_proj->forward(ctx, x);
            v = v_proj->forward(ctx, x);
        }

        x = ggml_ext_attention_ext(ctx, q, k, v, n_head, mask, false);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

#endif  // __SD_MODEL_COMMON_GGML_BLOCK_HPP__
