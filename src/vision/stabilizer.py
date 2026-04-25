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

    MIN_INLIERS = 12              # minimum RANSAC inliers to accept a homography
    MAX_FEATURES = 1000           # ORB features per frame
    MATCH_RATIO = 0.75            # Lowe ratio-test threshold
    MAX_TRANSLATION_FRAC = 0.35   # reject shifts larger than 35% of frame
    MIN_SCALE = 0.75              # reject extreme zoom-out
    MAX_SCALE = 1.35              # reject extreme zoom-in
    MAX_PERSPECTIVE_TERM = 0.0025 # reject strong projective skew
    STABLE_ALPHA = 0.25           # low-pass filter weight for homography smoothing
    REF_TOP_FRAC = 0.58           # use mostly-static upper scene for keypoints
    MAX_FALLBACK_FRAMES = 8       # reuse last valid transform for short dropouts

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.MAX_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._ref_kp = None
        self._ref_desc = None
        self._ref_shape: tuple[int, int] | None = None

        # last valid inverse homography (identity until a good one is found)
        self._last_H_inv: np.ndarray = np.eye(3, dtype=np.float64)
        self._has_valid_transform: bool = False
        self._fallback_frames: int = 0
        self._stable: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_reference(self, frame: np.ndarray) -> None:
        """Call exactly once on frame 0 before entering the main loop."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_mask = self._build_feature_mask(gray.shape[0], gray.shape[1])
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(gray, ref_mask)
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
        feature_mask = self._build_feature_mask(h, w)
        kp, desc = self._orb.detectAndCompute(gray, feature_mask)

        H_inv = self._estimate_inverse_homography(kp, desc)

        if H_inv is not None:
            self._last_H_inv = self._smooth_homography(H_inv)
            self._has_valid_transform = True
            self._fallback_frames = 0
            self._stable = True
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

        # Fail-open behavior: if homography is unstable, show the original frame
        # instead of warping with a stale transform (prevents "angled/black" output).
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

        if not self._is_homography_sane(H):
            return None

        return H  # already the inverse direction (current → ref)

    def _is_homography_sane(self, H: np.ndarray) -> bool:
        """Reject transforms that are likely to produce heavy perspective distortion."""
        if H.shape != (3, 3):
            return False
        if not np.isfinite(H).all():
            return False

        # Normalize to keep comparisons stable.
        if abs(H[2, 2]) < 1e-9:
            return False
        Hn = H / H[2, 2]

        # Projective terms too large => strong keystone/shear artifacts.
        if abs(Hn[2, 0]) > self.MAX_PERSPECTIVE_TERM or abs(Hn[2, 1]) > self.MAX_PERSPECTIVE_TERM:
            return False

        # Translation bounded by frame size.
        h, w = self._ref_shape if self._ref_shape is not None else (720, 1280)
        max_tx = w * self.MAX_TRANSLATION_FRAC
        max_ty = h * self.MAX_TRANSLATION_FRAC
        if abs(Hn[0, 2]) > max_tx or abs(Hn[1, 2]) > max_ty:
            return False

        # Approximate local scales from affine part.
        sx = float(np.hypot(Hn[0, 0], Hn[1, 0]))
        sy = float(np.hypot(Hn[0, 1], Hn[1, 1]))
        if not (self.MIN_SCALE <= sx <= self.MAX_SCALE):
            return False
        if not (self.MIN_SCALE <= sy <= self.MAX_SCALE):
            return False

        return True

    def _smooth_homography(self, H_inv: np.ndarray) -> np.ndarray:
        """Apply lightweight temporal smoothing to reduce per-frame warp jitter."""
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
        """Bias keypoints toward static background and away from moving road traffic."""
        mask = np.zeros((h, w), dtype=np.uint8)
        top_h = max(1, int(h * self.REF_TOP_FRAC))
        mask[:top_h, :] = 255
        return mask

    @staticmethod
    def _draw_status(frame: np.ndarray) -> None:
        pass  # status is drawn by the caller via is_stable property

    @property
    def is_stable(self) -> bool:
        return self._stable
