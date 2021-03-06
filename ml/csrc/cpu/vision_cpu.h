#pragma once
#include <torch/extension.h>

at::Tensor ROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale_x,
    const float spatial_scale_y,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlign_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale_x,
    const float spatial_scale_y,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale_x,
    const float spatial_scale_y,
    const int pooled_height,
    const int pooled_width);

at::Tensor ROIPool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale_x,
    const float spatial_scale_y,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

at::Tensor nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);
