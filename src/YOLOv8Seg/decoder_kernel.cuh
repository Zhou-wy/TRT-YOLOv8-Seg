/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-08 14:39:29
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-10 10:25:36
 */
#ifndef __DECODER_KERNEL_CUH
#define __DECODER_KERNEL_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include "yolov8_seg.hpp"


void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                           float confidence_threshold, float nms_threshold,
                           float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                           int NUM_BOX_ELEMENT, YOLOv8Seg::Task task, cudaStream_t stream);

void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                        int mask_width, int mask_height, unsigned char *mask_out,
                        int mask_dim, int out_width, int out_height, cudaStream_t stream);
#endif