/*
 * @description: 
 * @version: 
 * @Author: zwy
 * @Date: 2023-07-11 19:01:45
 * @LastEditors: zwy
 * @LastEditTime: 2023-09-06 13:55:05
 */

#include <vector>
#include <opencv2/opencv.hpp>


static const char *echargerlabels[] = {"Cable-1", "Cable-2"};
std::vector<cv::Scalar> echargercolors = {
    {160, 82, 45}, // Sienna
    {220, 20, 60}, // Crimson
};

static const char *coco_labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

std::vector<cv::Scalar> coco_colors = {
    {0, 0, 255},     // Red
    {0, 255, 0},     // Green
    {255, 0, 0},     // Bule
    {255, 255, 0},   // Cyan
    {0, 255, 255},   // Yellow
    {255, 0, 255},   // Magenta
    {128, 0, 0},     // Navy
    {0, 128, 0},     // Olive
    {0, 0, 128},     // Maroon
    {128, 128, 0},   // Teal
    {128, 0, 128},   // Purple
    {0, 128, 128},   // Gray
    {192, 192, 192}, // Silver
    {128, 128, 128}, // Aqua
    {255, 192, 203}, // Pink
    {255, 165, 0},   // Orange
    {255, 255, 255}, // White
    {0, 0, 0},       // Black
    {255, 255, 240}, // Ivory
    {165, 42, 42},   // Brown
    {0, 0, 205},     // MediumBlue
    {255, 250, 240}, // FloralWhite
    {255, 0, 255},   // Fuchsia
    {220, 20, 60},   // Crimson
    {0, 128, 0},     // Green
    {128, 0, 0},     // Maroon
    {255, 215, 0},   // Gold
    {138, 43, 226},  // BlueViolet
    {255, 105, 180}, // HotPink
    {160, 82, 45},   // Sienna
    {128, 128, 128}, // Gray
    {255, 69, 0},    // OrangeRed
    {128, 0, 128},   // Purple
    {0, 0, 139},     // DarkBlue
    {255, 250, 250}, // Snow
    {0, 139, 139},   // DarkCyan
    {70, 130, 180},  // SteelBlue
    {128, 128, 0},   // Olive
    {240, 248, 255}, // AliceBlue
    {255, 239, 213}, // PapayaWhip
    {255, 192, 203}, // Pink
    {0, 255, 255},   // Cyan
    {0, 255, 0},     // Green
    {255, 222, 173}, // NavajoWhite
    {218, 112, 214}, // Orchid
    {255, 0, 0},     // Blue
    {0, 0, 205},     // MediumBlue
    {238, 232, 170}, // PaleGoldenRod
    {75, 0, 130},    // Indigo
    {176, 224, 230}, // PowderBlue
    {127, 255, 212}, // Aquamarine
    {222, 184, 135}, // BurlyWood
    {255, 250, 205}, // LemonChiffon
    {255, 228, 196}, // Bisque
    {205, 92, 92},   // IndianRed
    {139, 0, 139},   // DarkMagenta
    {135, 206, 250}, // LightSkyBlue
    {250, 128, 114}, // Salmon
    {255, 105, 180}, // HotPink
    {154, 205, 50},  // YellowGreen
    {240, 230, 140}, // Khaki
    {255, 255, 0},   // Yellow
    {219, 112, 147}, // PaleVioletRed
    {255, 0, 255},   // Magenta
    {255, 215, 0},   // Gold
    {128, 0, 0},     // Maroon
    {0, 0, 128},     // Navy
    {0, 255, 255},   // Aqua
    {255, 228, 181}, // Moccasin
    {189, 183, 107}, // DarkKhaki
    {221, 160, 221}, // Plum
    {102, 205, 170}, // MediumAquamarine
    {173, 255, 47},  // GreenYellow
    {139, 69, 19},   // SaddleBrown
    {0, 0, 0},       // Black
    {255, 255, 255}, // White
};