import cv2
import numpy as np


def draw_overlay(frame, fusion_result, pose_result=None, yolo_score=None, pose_score=None, velocity_score=None, timestamp=None, **kwargs):
    h, w = frame.shape[:2]
    level = fusion_result.alert_level
    score = fusion_result.fused_score

    # Colors
    color_map = {
        "normal": (0, 200, 0),
        "possibly_suspicious": (0, 165, 255),
        "suspicious": (0, 0, 255)
    }
    color = color_map.get(level, (0, 200, 0))

    # Draw border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)

    # Draw score bar background
    bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    # Draw score bar fill
    fill_w = int(bar_w * score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Draw score text
    cv2.putText(frame, f'Score: {score:.2f}', (bar_x, bar_y + bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw alert level
    cv2.putText(frame, level.upper(), (bar_x, bar_y + bar_h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw individual scores
    y_offset = bar_y + bar_h + 75
    if yolo_score is not None:
        cv2.putText(frame, f'YOLO: {yolo_score:.2f}', (bar_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    if pose_score is not None:
        cv2.putText(frame, f'Pose: {pose_score:.2f}', (bar_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    if velocity_score is not None:
        cv2.putText(frame, f'Velocity: {velocity_score:.2f}', (bar_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    # Draw triggered rules
    if pose_result and pose_result.triggered_rules:
        for rule in pose_result.triggered_rules[:3]:
            cv2.putText(frame, f'! {rule}', (bar_x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_offset += 20

    return frame