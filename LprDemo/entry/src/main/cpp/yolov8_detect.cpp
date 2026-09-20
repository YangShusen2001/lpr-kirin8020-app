/**
 * YOLOv8 detector integration for HarmonyOS.
 * 
 * This file provides YOLOv8 support as an alternative to the current YOLOv5-face implementation.
 * Key differences from YOLOv5:
 * - Input size: 640x640 (vs 320x320 for YOLOv5)
 * - Output format: Different anchor-free detection head
 * - Better accuracy and speed on modern hardware
 */

#include "yolov8_detect.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace {

/**
 * YOLOv8 output structure: [batch, anchors, 84] where 84 = 4 bbox + 80 classes
 * For license plate detection, we only care about the first class (plate).
 */
struct YoloV8Output {
  std::vector<float> boxes;  // x1, y1, x2, y2 per anchor
  std::vector<float> scores; // confidence per anchor
  int num_anchors = 8400;    // Standard YOLOv8n anchor count
};

/**
 * Decode YOLOv8 output to bounding boxes.
 * YOLOv8 uses a different decoding scheme than YOLOv5.
 */
YoloV8Output DecodeYoloV8(const std::vector<float>& raw_output, int img_w, int img_h) {
  YoloV8Output output;
  output.num_anchors = raw_output.size() / 84;
  
  const float* data = raw_output.data();
  for (int i = 0; i < output.num_anchors; i++) {
    // YOLOv8 output format: [x1, y1, x2, y2, class_scores...]
    float x1 = data[i * 84 + 0];
    float y1 = data[i * 84 + 1];
    float x2 = data[i * 84 + 2];
    float y2 = data[i * 84 + 3];
    float score = data[i * 84 + 4];  // First class score (plate)
    
    if (score > 0.25f) {  // Confidence threshold
      output.boxes.push_back(x1);
      output.boxes.push_back(y1);
      output.boxes.push_back(x2);
      output.boxes.push_back(y2);
      output.scores.push_back(score);
    }
  }
  
  return output;
}

/**
 * NMS for YOLOv8 output.
 */
std::vector<std::vector<float>> YoloV8NMS(const YoloV8Output& output, float iou_thresh) {
  std::vector<size_t> indices(output.scores.size());
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }
  
  // Sort by score descending
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return output.scores[a] > output.scores[b];
  });
  
  std::vector<bool> kept(output.scores.size(), false);
  std::vector<std::vector<float>> final_boxes;
  
  for (size_t idx : indices) {
    if (kept[idx]) continue;
    
    kept[idx] = true;
    final_boxes.push_back({output.boxes[idx*4], output.boxes[idx*4+1], 
                           output.boxes[idx*4+2], output.boxes[idx*4+3], 
                           output.scores[idx]});
    
    // NMS with remaining boxes
    for (size_t j = idx + 1; j < indices.size(); j++) {
      if (kept[indices[j]]) continue;
      
      size_t idx_j = indices[j];
      // Calculate IoU
      float x1 = std::max(final_boxes.back()[0], output.boxes[idx_j*4]);
      float y1 = std::max(final_boxes.back()[1], output.boxes[idx_j*4+1]);
      float x2 = std::min(final_boxes.back()[2], output.boxes[idx_j*4+2]);
      float y2 = std::min(final_boxes.back()[3], output.boxes[idx_j*4+3]);
      
      float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
      float area1 = (final_boxes.back()[2] - final_boxes.back()[0]) * 
                    (final_boxes.back()[3] - final_boxes.back()[1]);
      float area2 = (output.boxes[idx_j*4+2] - output.boxes[idx_j*4]) * 
                    (output.boxes[idx_j*4+3] - output.boxes[idx_j*4+1]);
      float iou = inter / (area1 + area2 - inter);
      
      if (iou > iou_thresh) {
        kept[idx_j] = true;
      }
    }
  }
  
  return final_boxes;
}

}  // namespace

std::vector<std::vector<float>> DetectYoloV8(const LetterBoxed& lb, 
                                              const std::vector<float>& model_output,
                                              float conf_thresh, 
                                              float iou_thresh) {
  // Decode YOLOv8 output
  YoloV8Output decoded = DecodeYoloV8(model_output, lb.img.width, lb.img.height);
  
  // Apply NMS
  std::vector<std::vector<float>> nms_boxes = YoloV8NMS(decoded, iou_thresh);
  
  // Scale boxes back to original image coordinates
  std::vector<std::vector<float>> final_boxes;
  for (const auto& box : nms_boxes) {
    float x1 = (box[0] - lb.left) / lb.r;
    float y1 = (box[1] - lb.top) / lb.r;
    float x2 = (box[2] - lb.left) / lb.r;
    float y2 = (box[3] - lb.top) / lb.r;
    
    final_boxes.push_back({x1, y1, x2, y2, box[4]});
  }
  
  return final_boxes;
}
