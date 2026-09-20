#ifndef YOLOV8_DETECT_H
#define YOLOV8_DETECT_H

#include "lpr_pipeline.h"
#include <vector>

/**
 * Detect license plates using YOLOv8 model.
 * 
 * @param lb LetterBoxed image (already resized to 640x640)
 * @param model_output Raw model output tensor
 * @param conf_thresh Confidence threshold (default: 0.25)
 * @param iou_thresh IoU threshold for NMS (default: 0.5)
 * @return Vector of bounding boxes [x1, y1, x2, y2, score]
 */
std::vector<std::vector<float>> DetectYoloV8(const LetterBoxed& lb,
                                              const std::vector<float>& model_output,
                                              float conf_thresh = 0.25f,
                                              float iou_thresh = 0.5f);

#endif  // YOLOV8_DETECT_H
