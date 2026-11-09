#include "core/segment_weight_pipeline.h"

#include <cstdlib>
#include <utility>

#include "core/ggml_graph_cut.h"
#include "device_residency_manager.h"

namespace sd {
    static ggml_tensor* canonical_param(
        ggml_tensor* tensor,
        const std::unordered_set<const ggml_tensor*>& params) {
        for (ggml_tensor* current = tensor; current != nullptr; current = current->view_src) {
            if (params.find(current) != params.end()) {
                return current;
            }
        }
        return nullptr;
    }

    SegmentWeightPipeline::SegmentWeightPipeline(
        const std::shared_ptr<DeviceResidencyManager>& residency_manager,
        ggml_backend_t compute_backend,
        uintptr_t owner_id,
        ggml_cgraph* graph,
        const ggml_graph_cut::Plan& plan,
        const std::unordered_set<const ggml_tensor*>& params,
        bool enabled)
        : residency_manager_(residency_manager),
          compute_backend_(compute_backend),
          owner_id_(owner_id),
          enabled_(enabled && residency_manager != nullptr) {
        static const bool preload_env = [] {
            const char* value = std::getenv("SD_SEGMENT_PRELOAD");
            return value == nullptr || value[0] != '0';
        }();
        preload_enabled_ = enabled_ && preload_env;
        segment_params_.resize(plan.segments.size());
        for (size_t segment_index = 0; segment_index < plan.segments.size(); ++segment_index) {
            std::unordered_set<ggml_tensor*> seen;
            for (ggml_tensor* tensor :
                 ggml_graph_cut::param_tensors(graph, plan.segments[segment_index])) {
                ggml_tensor* param = canonical_param(tensor, params);
                if (param != nullptr && seen.insert(param).second) {
                    segment_params_[segment_index].push_back(param);
                }
            }
        }
    }

    SegmentWeightPipeline::~SegmentWeightPipeline() {
        segment_end();
        release_preloaded();
        clear();
    }

    std::vector<ggml_tensor*> SegmentWeightPipeline::next_missing_params(size_t segment_index,
                                                                         size_t next_segment) const {
        std::unordered_set<ggml_tensor*> active_params(segment_params_[segment_index].begin(),
                                                       segment_params_[segment_index].end());
        std::vector<ggml_tensor*> params;
        params.reserve(segment_params_[next_segment].size());
        for (ggml_tensor* param : segment_params_[next_segment]) {
            if (active_params.find(param) == active_params.end()) {
                params.push_back(param);
            }
        }
        return params;
    }

    void SegmentWeightPipeline::release_preloaded() {
        if (preloaded_params_.empty()) {
            return;
        }
        if (auto manager = residency_manager_.lock()) {
            manager->release_compute_backend_params(preloaded_params_);
        }
        preloaded_params_.clear();
    }

    bool SegmentWeightPipeline::can_preload_next(size_t segment_index) {
        if (!preload_enabled_ || !preloaded_params_.empty() || queued_segment_ != SIZE_MAX ||
            segment_index >= segment_params_.size()) {
            return false;
        }
        const size_t next_segment = next_parameter_segment(segment_index);
        auto manager              = residency_manager_.lock();
        if (next_segment == SIZE_MAX || manager == nullptr) {
            return false;
        }
        const auto params = next_missing_params(segment_index, next_segment);
        return !params.empty() && manager->can_prepare_params_during_compute(params) &&
               manager->inspect_compute_backend_params(params).missing_bytes > 0;
    }

    bool SegmentWeightPipeline::preload_next(size_t segment_index, const DeviceMemoryRequest& request) {
        const size_t next_segment = next_parameter_segment(segment_index);
        auto manager              = residency_manager_.lock();
        if (next_segment == SIZE_MAX || manager == nullptr) {
            return false;
        }
        auto params                         = next_missing_params(segment_index, next_segment);
        DeviceMemoryRequest backend_request = request;
        backend_request.compute_backend     = compute_backend_;
        backend_request.owner_id            = owner_id_;
        std::vector<ggml_tensor*> protected_params = segment_params_[segment_index];
        protected_params.insert(protected_params.end(), params.begin(), params.end());
        if (!manager->ensure_compute_backend_capacity(backend_request, params, preferred_eviction_order(),
                                                      protected_params) ||
            !manager->prepare_params(params)) {
            return false;
        }
        preloaded_params_ = std::move(params);
        return true;
    }

    size_t SegmentWeightPipeline::next_parameter_segment(size_t segment_index) const {
        for (size_t next = segment_index + 1; next < segment_params_.size(); ++next) {
            if (!segment_params_[next].empty()) {
                return next;
            }
        }
        return SIZE_MAX;
    }

    std::vector<std::vector<ggml_tensor*>> SegmentWeightPipeline::preferred_eviction_order() const {
        return {segment_params_.rbegin(), segment_params_.rend()};
    }

    void SegmentWeightPipeline::disable() {
        clear();
        enabled_ = false;
    }

    void SegmentWeightPipeline::activate(size_t segment_index) {
        if (!enabled_ || queued_segment_ == SIZE_MAX || queued_segment_ != segment_index) {
            return;
        }

        auto manager = residency_manager_.lock();
        if (manager == nullptr ||
            !manager->activate_prefetched_params(owner_id_, queued_params_)) {
            disable();
            return;
        }
        queued_params_.clear();
        queued_segment_ = SIZE_MAX;
    }

    bool SegmentWeightPipeline::ensure_segment_capacity(
        size_t segment_index,
        const std::vector<DeviceMemoryRequest>& requests) {
        if (segment_index >= segment_params_.size()) {
            return false;
        }
        auto manager = residency_manager_.lock();
        if (manager == nullptr) {
            return segment_params_[segment_index].empty();
        }
        std::vector<ggml_tensor*> protected_params = segment_params_[segment_index];
        protected_params.insert(protected_params.end(), queued_params_.begin(), queued_params_.end());
        for (const auto& request : requests) {
            if (!manager->ensure_compute_backend_capacity(request, segment_params_[segment_index],
                                                          preferred_eviction_order(), protected_params)) {
                return false;
            }
        }
        return true;
    }

    bool SegmentWeightPipeline::segment_start(size_t segment_index, const std::function<bool()>& ensure_capacity) {
        GGML_ASSERT(pinned_params_.empty());
        activate(segment_index);
        if (!ensure_capacity()) {
            return false;
        }
        auto manager = residency_manager_.lock();
        if (manager == nullptr) {
            return segment_params_[segment_index].empty();
        }
        if (!manager->prepare_params(segment_params_[segment_index])) {
            return false;
        }
        pinned_params_ = segment_params_[segment_index];
        // prepare_params pinned the preloaded weights again for this segment.
        release_preloaded();
        return true;
    }

    void SegmentWeightPipeline::segment_end() {
        if (auto manager = residency_manager_.lock()) {
            manager->release_compute_backend_params(pinned_params_);
        }
        pinned_params_.clear();
    }

    void SegmentWeightPipeline::enqueue_next(
        size_t segment_index,
        const DeviceMemoryRequest& request) {
        if (!enabled_ || queued_segment_ != SIZE_MAX) {
            return;
        }

        const size_t next_segment = next_parameter_segment(segment_index);
        if (next_segment == SIZE_MAX) {
            return;
        }

        std::vector<ggml_tensor*> params = next_missing_params(segment_index, next_segment);
        if (params.empty()) {
            return;
        }

        auto manager = residency_manager_.lock();
        if (manager == nullptr) {
            disable();
            return;
        }
        const WeightResidencyInfo residency =
            manager->inspect_compute_backend_params(params);
        if (residency.missing_bytes == 0) {
            return;
        }
        if (!residency.async_prefetch_supported) {
            disable();
            return;
        }

        DeviceMemoryRequest backend_request        = request;
        backend_request.compute_backend            = compute_backend_;
        backend_request.owner_id                   = owner_id_;
        std::vector<ggml_tensor*> protected_params = segment_params_[segment_index];
        protected_params.insert(protected_params.end(), params.begin(), params.end());
        if (!manager->ensure_compute_backend_capacity(backend_request,
                                                      params,
                                                      preferred_eviction_order(),
                                                      protected_params)) {
            disable();
            return;
        }
        switch (manager->prepare_prefetch_params(owner_id_, params)) {
            case WeightPrefetchResult::Scheduled:
                queued_params_  = std::move(params);
                queued_segment_ = next_segment;
                return;
            case WeightPrefetchResult::AlreadyResident:
                return;
            case WeightPrefetchResult::Unsupported:
                disable();
                return;
            case WeightPrefetchResult::Failed:
                disable();
                return;
        }
        disable();
    }

    void SegmentWeightPipeline::submit_next() {
        if (queued_segment_ != SIZE_MAX) {
            if (auto manager = residency_manager_.lock()) {
                manager->submit_prefetched_params(owner_id_);
            }
        }
    }

    void SegmentWeightPipeline::clear() {
        if (auto manager = residency_manager_.lock()) {
            manager->clear_prefetched_params(owner_id_);
        }
        queued_params_.clear();
        queued_segment_ = SIZE_MAX;
    }
}
