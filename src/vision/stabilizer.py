from __future__ import annotations
import os
import cv2
import numpy as np
class VideoStabilizer:
    MIN_INLIERS = 12
    MAX_FEATURES = 1000
    MATCH_RATIO = 0.75
    MAX_TRANSLATION_FRAC = 0.35
    MIN_SCALE = 0.75
    MAX_SCALE = 1.35
    MAX_PERSPECTIVE_TERM = 0.0025
    STABLE_ALPHA = 0.25
    REF_TOP_FRAC = 0.58
    MAX_FALLBACK_FRAMES = 8
    LK_MIN_VALID = 20
    LK_GRID_STEP = 40
    LK_MARGIN = 20
    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.MAX_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._ref_grey: np.ndarray | None = None
        self._ref_kp = None
        self._ref_desc = None
        self._ref_shape: tuple[int, int] | None = None
        self._last_H_inv: np.ndarray = np.eye(3, dtype=np.float64)
        self._has_valid_transform: bool = False
        self._fallback_frames: int = 0
        self._stable: bool = True
        self._last_method: str = "none"
        self._last_orb_good_count: int = 0
        self._orb_min_matches: int = int(os.getenv("ORB_MIN_MATCHES", str(self.MIN_INLIERS)))
    def init_reference(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_mask = self._build_feature_mask(gray.shape[0], gray.shape[1])
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(gray, ref_mask)
        self._ref_shape = gray.shape
        self._ref_grey = gray.copy()
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        if self._ref_desc is None:
            return frame
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_mask = self._build_feature_mask(h, w)
        kp, desc = self._orb.detectAndCompute(gray, feature_mask)
        H_inv, method = self._estimate_homography_with_fallback(gray, kp, desc)
        if H_inv is not None:
            self._last_H_inv = self._smooth_homography(H_inv)
            self._has_valid_transform = True
            self._fallback_frames = 0
            self._stable = True
            self._last_method = method
        else:
            self._stable = False
        if H_inv is None and self._has_valid_transform and self._fallback_frames < self.MAX_FALLBACK_FRAMES:
            self._fallback_frames += 1
            stabilised = cv2.warpPerspective(
                frame, self._last_H_inv, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            self._draw_status(stabilised)
            return stabilised
        if not self._stable:
            self._draw_status(frame)
            return frame
        stabilised = cv2.warpPerspective(
            frame, self._last_H_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self._draw_status(stabilised)
        return stabilised
    @property
    def last_method(self) -> str:
        return self._last_method
    def _estimate_homography_with_fallback(
        self, curr_grey: np.ndarray, kp, desc
    ) -> tuple[np.ndarray | None, str]:
        H_orb = self._estimate_inverse_homography(kp, desc)
        orb_good = self._last_orb_good_count
        if H_orb is not None and orb_good >= self._orb_min_matches:
            return H_orb, "orb"
        if self._ref_grey is not None:
            H_lk = self._lucas_kanade_fallback(self._ref_grey, curr_grey)
            if H_lk is not None:
                return H_lk, "lucas_kanade"
        if H_orb is not None:
            return H_orb, "orb_low_matches"
        return None, "identity"
    def _lucas_kanade_fallback(
        self, prev_grey: np.ndarray, curr_grey: np.ndarray
    ) -> np.ndarray | None:
        h, w = prev_grey.shape
        m = self.LK_MARGIN
        s = self.LK_GRID_STEP
        gx, gy = np.mgrid[m:w - m:s, m:h - m:s]
        p0 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
        if len(p0) < self.LK_MIN_VALID:
            return None
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_grey, curr_grey, p0, None, **lk_params)
        if p1 is None or status is None:
            return None
        valid = status.flatten() == 1
        if valid.sum() < self.LK_MIN_VALID:
            return None
        H, mask = cv2.findHomography(p1[valid], p0[valid], cv2.RANSAC, 5.0)
        if H is None:
            return None
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.LK_MIN_VALID:
            return None
        if not self._is_homography_sane(H):
            return None
        return H
    def _estimate_inverse_homography(
        self, kp, desc
    ) -> np.ndarray | None:
        self._last_orb_good_count = 0
        if desc is None or len(kp) < self.MIN_INLIERS:
            return None
        matches = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.MATCH_RATIO * n.distance:
                    good.append(m)
        self._last_orb_good_count = len(good)
        if len(good) < self.MIN_INLIERS:
            return None
        src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.MIN_INLIERS:
            return None
        if not self._is_homography_sane(H):
            return None
        return H
    def _is_homography_sane(self, H: np.ndarray) -> bool:
        if H.shape != (3, 3):
            return False
        if not np.isfinite(H).all():
            return False
        if abs(H[2, 2]) < 1e-9:
            return False
        Hn = H / H[2, 2]
        if abs(Hn[2, 0]) > self.MAX_PERSPECTIVE_TERM or abs(Hn[2, 1]) > self.MAX_PERSPECTIVE_TERM:
            return False
        h, w = self._ref_shape if self._ref_shape is not None else (720, 1280)
        max_tx = w * self.MAX_TRANSLATION_FRAC
        max_ty = h * self.MAX_TRANSLATION_FRAC
        if abs(Hn[0, 2]) > max_tx or abs(Hn[1, 2]) > max_ty:
            return False
        sx = float(np.hypot(Hn[0, 0], Hn[1, 0]))
        sy = float(np.hypot(Hn[0, 1], Hn[1, 1]))
        if not (self.MIN_SCALE <= sx <= self.MAX_SCALE):
            return False
        if not (self.MIN_SCALE <= sy <= self.MAX_SCALE):
            return False
        return True
    def _smooth_homography(self, H_inv: np.ndarray) -> np.ndarray:
        if abs(H_inv[2, 2]) < 1e-9:
            return self._last_H_inv
        current = H_inv / H_inv[2, 2]
        if not self._has_valid_transform:
            return current
        prev = self._last_H_inv
        if abs(prev[2, 2]) < 1e-9:
            prev = np.eye(3, dtype=np.float64)
        else:
            prev = prev / prev[2, 2]
        blended = (1.0 - self.STABLE_ALPHA) * prev + self.STABLE_ALPHA * current
        if abs(blended[2, 2]) < 1e-9:
            blended[2, 2] = 1.0
        blended = blended / blended[2, 2]
        if self._is_homography_sane(blended):
            return blended
        return current
    def _build_feature_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        top_h = max(1, int(h * self.REF_TOP_FRAC))
        mask[:top_h, :] = 255
        return mask
    @staticmethod
    def _draw_status(frame: np.ndarray) -> None:
        pass
    @property
    def is_stable(self) -> bool:
        return self._stable
