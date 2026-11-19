"""
Decide whether two photos show the same object or place.

Purely mathematical pipeline (no neural networks), built to tolerate changes in
viewpoint and lighting:

  1. Preprocessing : common scale, automatic gamma, conditional Non-Local Means
                     denoising, CLAHE
  2. Two channels  : (a) normalized intensity
                     (b) phase congruency (Kovesi, log-Gabor filters), which is
                         invariant to brightness and contrast by construction
  3. ASIFT         : simulates camera tilts (t = sqrt(2)^k) and rotations, runs
                     SIFT on every simulated view and maps keypoints back to the
                     original image coordinates
  4. RootSIFT      : L1-normalized, square-rooted descriptors (Hellinger distance)
  5. Matching      : mutual nearest neighbour + Lowe ratio test that ignores ASIFT
                     duplicates of the same physical point
  6. Verification  : fundamental matrix (true 3D epipolar geometry) estimated
                     with USAC MAGSAC++

CLI usage: python -m app.comparison img1.jpg img2.jpg [--fast] [--details]
Runs fully offline.
"""

import argparse

import cv2
import numpy as np
from pydantic import BaseModel

# ---------------------------------------------------------------- parameters

WORKING_HEIGHT = 480          # px, both images are resized to this height
INLIER_THRESHOLD = 25         # minimum inliers for "same place" (calibrate on your data)
INLIER_RATIO_THRESHOLD = 0.20 # minimum share of candidate matches that must be inliers
LOWE_RATIO = 0.85
RANSAC_THRESHOLD_PX = 1.5
DUPLICATE_RADIUS_PX = 6.0     # keypoints closer than this are treated as the same point
MIN_POINTS_FOR_GEOMETRY = 12


class ImageDecodeError(ValueError):
    pass


class ComparisonResult(BaseModel):
    matches: int
    inliers: int
    inlier_ratio: float
    score: float
    is_match: bool


# ---------------------------------------------------------------- 1. preprocessing

def decode_image(data: bytes, name: str) -> np.ndarray:
    if not data:
        raise ImageDecodeError(f"Cannot read {name}")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageDecodeError(f"Cannot read {name}")
    return image


def scale(image: np.ndarray, height: int = WORKING_HEIGHT) -> np.ndarray:
    factor = height / image.shape[0]
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Automatic gamma (pulls the mean towards mid-gray) + denoising + CLAHE."""
    mean = np.clip(image.mean() / 255.0, 1e-3, 1 - 1e-3)
    gamma = np.clip(np.log(0.5) / np.log(mean), 0.4, 2.5)  # gamma < 1 brightens dark photos
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    image = cv2.LUT(image, lut)

    blurred = cv2.GaussianBlur(image, (0, 0), 1.0).astype(np.float32)
    noise = np.std(image.astype(np.float32) - blurred)
    if noise > 4:  # denoise only when needed (e.g. high-ISO night shots)
        image = cv2.fastNlMeansDenoising(
            image, None, h=float(min(15, 2 * noise)), templateWindowSize=7, searchWindowSize=21
        )
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)


# ---------------------------------------------------------------- 2. phase congruency

def phase_congruency(
    image: np.ndarray,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength: float = 3,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    eps: float = 1e-4,
    k: float = 2.0,
) -> np.ndarray:
    """
    Phase congruency (Kovesi 1999/2003) with log-Gabor filters in the frequency domain.

    PC(x) = sum_o max(E_o(x) - T_o, 0) / (sum_o sum_s A_so(x) + eps)

    E = local energy (phase alignment), A = amplitudes, T = estimated noise threshold.
    The result lies in [0, 1] and does NOT depend on brightness or contrast.
    """
    f = image.astype(np.float64)
    rows, cols = f.shape
    spectrum = np.fft.fft2(f)

    y, x = np.mgrid[-(rows // 2):rows - rows // 2, -(cols // 2):cols - cols // 2]
    x = x / cols
    y = y / rows
    radius = np.fft.ifftshift(np.sqrt(x**2 + y**2))
    radius[0, 0] = 1.0
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    # low-pass filter to suppress the corners of the spectrum
    low_pass = 1.0 / (1.0 + (radius / 0.45) ** 30)

    log_gabor = []
    for s in range(n_scales):
        wavelength = min_wavelength * mult**s
        centre_frequency = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / centre_frequency)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
        lg *= low_pass
        lg[0, 0] = 0
        log_gabor.append(lg)

    # expected sum of amplitudes across scales relative to the smallest scale
    amplitude_sum_factor = sum((1.0 / mult) ** s for s in range(n_scales))

    pc_total = np.zeros_like(f)
    amplitude_total = np.zeros_like(f)
    for o in range(n_orientations):
        angle = o * np.pi / n_orientations
        ds = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
        dc = cos_theta * np.cos(angle) + sin_theta * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * n_orientations / 2, np.pi)
        spread = (np.cos(dtheta) + 1) / 2  # raised-cosine angular window

        sum_even = np.zeros_like(f)
        sum_odd = np.zeros_like(f)
        sum_amplitude = np.zeros_like(f)
        tau = 0.0
        for s in range(n_scales):
            response = np.fft.ifft2(spectrum * log_gabor[s] * spread)
            amplitude = np.abs(response)
            sum_even += response.real
            sum_odd += response.imag
            sum_amplitude += amplitude
            if s == 0:
                tau = np.median(amplitude) / np.sqrt(np.log(4))  # Rayleigh noise estimate

        energy = np.sqrt(sum_even**2 + sum_odd**2)

        # Kovesi noise threshold T: mean + k * std of the noise energy
        estimated_sum_amplitude = tau * amplitude_sum_factor
        estimated_mean = estimated_sum_amplitude * np.sqrt(np.pi / 2)
        estimated_sigma = estimated_sum_amplitude * np.sqrt((4 - np.pi) / 2)
        threshold = estimated_mean + k * estimated_sigma

        pc_total += np.maximum(energy - threshold, 0)
        amplitude_total += sum_amplitude

    return np.clip(pc_total / (amplitude_total + eps), 0, 1)


def phase_channel(image: np.ndarray) -> np.ndarray:
    pc = cv2.GaussianBlur(phase_congruency(image), (0, 0), 0.8)
    return cv2.normalize(pc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------- 3+4. ASIFT + RootSIFT

def affine_view(image: np.ndarray, tilt: float, phi: float):
    """Simulate a tilted camera: rotate by phi, then compress along x by the tilt factor."""
    h, w = image.shape
    affine = np.float32([[1, 0, 0], [0, 1, 0]])
    mask = np.full((h, w), 255, np.uint8)

    if phi != 0.0:
        a = np.deg2rad(phi)
        c, s = np.cos(a), np.sin(a)
        rotation = np.float32([[c, -s], [s, c]])
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) @ rotation.T
        x, y, ww, hh = cv2.boundingRect(corners.reshape(1, -1, 2))
        affine = np.hstack([rotation, np.float32([[-x], [-y]])])
        image = cv2.warpAffine(image, affine, (ww, hh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, affine, (ww, hh), flags=cv2.INTER_NEAREST)

    if tilt != 1.0:
        sigma = 0.8 * np.sqrt(tilt * tilt - 1)  # anti-aliasing before subsampling
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=0.01)
        image = cv2.resize(image, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        affine[0] /= tilt

    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return image, mask, cv2.invertAffineTransform(affine)


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = descriptors / (np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(descriptors).astype(np.float32)


def asift(image: np.ndarray, sift, fast: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx2 keypoints in original coordinates, Nx128 RootSIFT descriptors)."""
    tilts = [1.0, np.sqrt(2), 2.0] if fast else [1.0, np.sqrt(2), 2.0, 2 * np.sqrt(2)]
    h, w = image.shape
    points, descriptors = [], []
    for tilt in tilts:
        angles = [0.0] if tilt == 1.0 else np.arange(0, 180, 72.0 / tilt)
        for phi in angles:
            view, mask, inverse = affine_view(image, tilt, phi)
            keypoints, desc = sift.detectAndCompute(view, mask)
            if desc is None:
                continue
            p = np.float32([kp.pt for kp in keypoints]) @ inverse[:, :2].T + inverse[:, 2]
            inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            points.append(p[inside])
            descriptors.append(rootsift(desc[inside]))
    if not points:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.vstack(points), np.vstack(descriptors)


# ---------------------------------------------------------------- 5. matching

def unique_pairs(points_a: np.ndarray, points_b: np.ndarray, cell: float = 4.0) -> np.ndarray:
    """Indices of pairs kept after collapsing duplicates to one per 4 px cell in both images."""
    key = np.hstack([np.round(points_a / cell), np.round(points_b / cell)])
    _, unique = np.unique(key, axis=0, return_index=True)
    return np.sort(unique)


def match(points_a, desc_a, points_b, desc_b, ratio=LOWE_RATIO, k=8,
          duplicate_radius=DUPLICATE_RADIUS_PX) -> np.ndarray:
    """
    Mutual nearest neighbour + Lowe ratio test adapted to ASIFT: the second neighbour
    must be a spatially DIFFERENT point (otherwise it is just a copy of the same point
    detected in another simulated view).
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), np.int32)

    index_b = cv2.flann_Index(desc_b, dict(algorithm=1, trees=5))
    neighbours, sq_dist = index_b.knnSearch(desc_a, min(k, len(desc_b)), params=dict(checks=64))
    index_a = cv2.flann_Index(desc_a, dict(algorithm=1, trees=5))
    back, _ = index_a.knnSearch(desc_b, 1, params=dict(checks=64))
    neighbours = neighbours.astype(np.intp)
    back = back.astype(np.intp)

    best = neighbours[:, 0]
    # first neighbour that lies far enough from the best one to be a different point
    far = np.linalg.norm(points_b[neighbours[:, 1:]] - points_b[best][:, None, :], axis=2) > duplicate_radius
    has_second = far.any(axis=1)
    second = 1 + far.argmax(axis=1)
    rows = np.arange(len(desc_a))
    d1 = np.sqrt(sq_dist[:, 0])
    d2 = np.sqrt(sq_dist[rows, second])
    passes_ratio = has_second & (d1 < ratio * d2)

    # spatial mutual check: the best match of b_j in A must be (a copy of) a_i
    mutual = np.linalg.norm(points_a[back[best, 0]] - points_a, axis=1) <= duplicate_radius

    keep = np.flatnonzero(passes_ratio & mutual)
    if not len(keep):
        return np.zeros((0, 2), np.int32)
    pairs = np.column_stack([keep, best[keep]]).astype(np.int32)
    return pairs[unique_pairs(points_a[pairs[:, 0]], points_b[pairs[:, 1]])]


# ---------------------------------------------------------------- 6. geometric verification

def count_inliers(points_a: np.ndarray, points_b: np.ndarray) -> int:
    if len(points_a) < MIN_POINTS_FOR_GEOMETRY:
        return 0
    _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.USAC_MAGSAC,
                                     RANSAC_THRESHOLD_PX, 0.9999, 20000)
    return int(mask.sum()) if mask is not None else 0


# ---------------------------------------------------------------- pipeline

def count_matches(image_a: np.ndarray, image_b: np.ndarray, fast: bool = False,
                  details: bool = False) -> tuple[int, int]:
    image_a = normalize_illumination(scale(image_a))
    image_b = normalize_illumination(scale(image_b))
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)

    all_a, all_b = [], []
    channels = (
        ("intensity", image_a, image_b),
        ("phase", phase_channel(image_a), phase_channel(image_b)),
    )
    for name, channel_a, channel_b in channels:
        points_a, desc_a = asift(channel_a, sift, fast)
        points_b, desc_b = asift(channel_b, sift, fast)
        pairs = match(points_a, desc_a, points_b, desc_b)
        if details:
            print(f"  [{name}] keypoints {len(points_a)}/{len(points_b)}, candidate matches {len(pairs)}")
        all_a.append(points_a[pairs[:, 0]])
        all_b.append(points_b[pairs[:, 1]])

    matched_a, matched_b = np.vstack(all_a), np.vstack(all_b)
    if len(matched_a):
        keep = unique_pairs(matched_a, matched_b)
        matched_a, matched_b = matched_a[keep], matched_b[keep]
    return len(matched_a), count_inliers(matched_a, matched_b)


def build_result(matches: int, inliers: int) -> ComparisonResult:
    ratio = inliers / matches if matches else 0.0
    score = min(1.0, inliers / (2 * INLIER_THRESHOLD)) * min(1.0, ratio / INLIER_RATIO_THRESHOLD)
    return ComparisonResult(
        matches=matches,
        inliers=inliers,
        inlier_ratio=ratio,
        score=score,
        is_match=inliers >= INLIER_THRESHOLD and ratio >= INLIER_RATIO_THRESHOLD,
    )


def compare_images(data_a: bytes, data_b: bytes, fast: bool = False) -> ComparisonResult:
    image_a = decode_image(data_a, "image_a")
    image_b = decode_image(data_b, "image_b")
    return build_result(*count_matches(image_a, image_b, fast))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two photos (same place/object?)")
    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument("--fast", action="store_true", help="fewer ASIFT views (viewpoint changes < ~60°)")
    parser.add_argument("--details", action="store_true", help="print per-channel statistics")
    args = parser.parse_args()

    images = []
    for path in (args.img1, args.img2):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            parser.error(f"Cannot read {path}")
        images.append(image)

    result = build_result(*count_matches(images[0], images[1], args.fast, args.details))
    print(f"Candidate matches: {result.matches}, geometric inliers: {result.inliers} "
          f"({result.inlier_ratio:.0%}), score: {result.score:.2f}")
    print("SAME PLACE/OBJECT" if result.is_match else "Probably different")


if __name__ == "__main__":
    main()
