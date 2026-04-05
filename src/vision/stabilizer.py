from __future__ import annotations

import cv2
import numpy as np


class VideoStabilizer:
    """
    ORB feature-based video stabilisation.

    On the first frame a reference grey image is captured from the full frame
    and ORB keypoints / descriptors are extracted.  For every subsequent frame
    a homography H is estimated between the reference and the current frame;
    the *inverse* of H is then applied via warpPerspective so that the scene
    is warped back to the reference perspective.

    If homography estimation fails (too few inliers) the last good transform
    is reused ("fallback").  A small overlay text indicates the current
    stabilisation state.
    """

    MIN_INLIERS = 8          # minimum RANSAC inliers to accept a homography
    MAX_FEATURES = 1000      # ORB features per frame
    MATCH_RATIO = 0.75       # Lowe ratio-test threshold

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.MAX_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._ref_kp = None
        self._ref_desc = None
        self._ref_shape: tuple[int, int] | None = None

        # last valid inverse homography (identity until a good one is found)
        self._last_H_inv: np.ndarray = np.eye(3, dtype=np.float64)
        self._stable: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_reference(self, frame: np.ndarray) -> None:
        """Call exactly once on frame 0 before entering the main loop."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(gray, None)
        self._ref_shape = gray.shape  # (h, w)

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a stabilised copy of *frame*.  The frame is warped so its
        perspective matches the reference frame.  If stabilisation is not yet
        initialised, the frame is returned unchanged.
        """
        if self._ref_desc is None:
            return frame

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)

        H_inv = self._estimate_inverse_homography(kp, desc)

        if H_inv is not None:
            self._last_H_inv = H_inv
            self._stable = True
        else:
            self._stable = False  # fall back to last known good transform

        stabilised = cv2.warpPerspective(
            frame, self._last_H_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self._draw_status(stabilised)
        return stabilised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_inverse_homography(
        self, kp, desc
    ) -> np.ndarray | None:
        if desc is None or len(kp) < self.MIN_INLIERS:
            return None

        matches = self._matcher.knnMatch(self._ref_desc, desc, k=2)

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.MATCH_RATIO * n.distance:
                    good.append(m)

        if len(good) < self.MIN_INLIERS:
            return None

        src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # H maps ref → current; we need the inverse (current → ref)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.MIN_INLIERS:
            return None

        return H  # already the inverse direction (current → ref)

    @staticmethod
    def _draw_status(frame: np.ndarray) -> None:
        pass  # status is drawn by the caller via is_stable property

    @property
    def is_stable(self) -> bool:
        return self._stable
