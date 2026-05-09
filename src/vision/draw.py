from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.42
_FONT_THICK = 1

_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (50, 220, 50),
    1: (0, 165, 255),
    2: (255, 80, 0),
    3: (200, 0, 200),
    5: (0, 200, 200),
    7: (0, 0, 230),
}

_CLASS_LABELS: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _put_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    (tw, th), bl = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICK)
    pad = 4
    lx1, ly1 = x, max(0, y - th - bl - pad)
    lx2, ly2 = x + tw + pad * 2, y
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), bg_color, -1)
    cv2.putText(
        frame, text, (lx1 + pad, ly2 - bl),
        _FONT, _FONT_SCALE, text_color, _FONT_THICK, cv2.LINE_AA,
    )


def _class_color(cls: int, override: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    if override is not None:
        return override
    return _CLASS_COLORS.get(cls, (180, 180, 180))


def draw_box(
    frame: np.ndarray,
    box: Tuple,
    label: str,
    color: Tuple[int, int, int] = (20, 210, 20),
    cls: int = -1,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    draw_color = _class_color(cls, override=None) if cls in _CLASS_COLORS else color
    if color != (20, 210, 20):
        draw_color = color
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
    dark = tuple(max(0, int(c * 0.55)) for c in draw_color)
    _put_label(frame, label.upper(), x1, y1, dark, (240, 240, 240))


def draw_segmentation_mask(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    cls: int = -1,
    alpha: float = 0.35,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    if mask is None or mask.size == 0:
        return
    fh, fw = frame.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (fh, fw):
        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

    fill_color = color if color is not None else _class_color(cls)
    overlay = frame.copy()
    overlay[mask > 0] = fill_color
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        darker = tuple(max(0, int(c * 0.7)) for c in fill_color)
        cv2.drawContours(frame, contours, -1, darker, 1, cv2.LINE_AA)


def draw_all_masks(
    frame: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    masks: list,
    alpha: float = 0.35,
) -> None:
    for pass_cls in (0, -1):
        for box, cls, mask in zip(boxes, classes, masks):
            c = int(cls)
            if pass_cls == 0 and c != 0:
                continue
            if pass_cls == -1 and c == 0:
                continue
            draw_segmentation_mask(frame, mask, cls=c, alpha=alpha)
