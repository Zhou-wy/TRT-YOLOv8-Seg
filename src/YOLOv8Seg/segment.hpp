/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-06 12:17:57
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-11 13:08:17
 */
#ifndef __SEGMENT_HPP
#define __SEGMENT_HPP

#include <memory>
#include "../TrtLib/common/cuda_tools.cuh"

struct InstanceSegmentMap
{
    int width = 0, height = 0;     // width % 8 == 0
    uint8_t *data = nullptr; // is width * height memory

    InstanceSegmentMap(int _width, int _height);
    virtual ~InstanceSegmentMap();
};

struct Box
{
    float left, top, right, bottom, confidence;
    int class_label;
    std::shared_ptr<InstanceSegmentMap> seg; // valid only in segment task

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label) {}
};

#endif