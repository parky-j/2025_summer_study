# batch_make_mtf.py
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# =========================
# 0) 설정/유틸
# =========================
LMK_RE = re.compile(r"landmark_(\d+)_(x|y|z)$")

def find_axis_columns(df, axis):
    """axis in {'x','y','z'}에 해당하는 landmark_*_axis 컬럼을 ID 기준 정렬"""
    cols = []
    for c in df.columns:
        m = LMK_RE.match(c)
        if m and m.group(2) == axis:
            lid = int(m.group(1))
            cols.append((lid, c))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]

def interpolate_and_impute(X):
    """열별 선형 보간 → 평균 대체 → 잔여 NaN은 0 대체"""
    df = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    if df.isna().values.any():
        df = df.fillna(df.mean(axis=0))
    return df.fillna(0.0).to_numpy(dtype=np.float64)

def pca_first_component(time_by_feat):
    """SVD 기반 PCA 1축 시계열 반환"""
    X = time_by_feat
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v1 = Vt[0]
    z = Xc @ v1
    return z

def quantize_by_quantiles(s, Q):
    """분위수 기반 Q-분할 양자화 → bin index(0..Q-1)"""
    s = np.asarray(s, dtype=np.float64)
    if np.allclose(s.std(), 0.0):
        return np.zeros_like(s, dtype=np.int32)
    qs = np.linspace(0.0, 1.0, Q + 1)
    edges = np.quantile(s, qs)
    if np.unique(edges).size < edges.size:
        smin, smax = float(s.min()), float(s.max())
        if smin == smax:
            return np.zeros_like(s, dtype=np.int32)
        edges = np.linspace(smin, smax, Q + 1)
    bins = np.searchsorted(edges, s, side="right") - 1
    return np.clip(bins, 0, Q - 1).astype(np.int32)

def mtf_from_series(s, Q=32, smooth=1e-12):
    """
    1D 시계열 s → MTF 이미지 (T x T)
    M[i,j] = A[q_i, q_j], A는 전이확률행렬
    """
    s = np.asarray(s, dtype=np.float64)
    T = s.shape[0]
    q = quantize_by_quantiles(s, Q=Q)
    A = np.zeros((Q, Q), dtype=np.float64)
    if T >= 2:
        src, dst = q[:-1], q[1:]
        np.add.at(A, (src, dst), 1.0)
    A = A + smooth
    A = A / A.sum(axis=1, keepdims=True)
    M = A[q[:, None], q[None, :]]  # (T, T)
    return M

def resize_to(M, size=224):
    """bilinear 리사이즈 + [0,1] 정규화, float32 반환"""
    M = np.asarray(M, dtype=np.float32)
    img = Image.fromarray(M)  # mode 'F'
    img = img.resize((size, size), resample=Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    mmin, mmax = float(out.min()), float(out.max())
    if mmax > mmin:
        out = (out - mmin) / (mmax - mmin)
    else:
        out = np.zeros_like(out, dtype=np.float32)
    return out

def map_out_path(csv_path: Path, src_segment="split_eye_landmark", dst_segment="split_eye_landmark_npy"):
    """
    경로 세그먼트 'split_eye_landmark' → 'split_eye_landmark_npy' 치환.
    파일명은 동일, 확장자만 .npy
    """
    parts = list(csv_path.parts)
    try:
        idx = parts.index(src_segment)
    except ValueError:
        # 세그먼트가 없으면 문자열 치환으로 폴백 (안전장치)
        replaced = Path(str(csv_path).replace(src_segment, dst_segment))
        return replaced.with_suffix(".npy")
    parts[idx] = dst_segment
    out_dir = Path(*parts[:-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (csv_path.stem + ".npy")

def maybe_downsample_series(s, max_len=None):
    """MTF 비용(T^2) 제어를 위해 시계열을 균등간격 다운샘플 (선택사항)"""
    if max_len is None or len(s) <= max_len:
        return s
    idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
    return s[idx]

def build_3ch_mtf_from_csv(csv_path, Q=32, size=224, drop_nan_ratio=0.5, max_len=None):
    """
    csv_path: 랜드마크 CSV (행=시간)
    반환: (3, size, size) float32
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    channels = []
    for axis in ["x", "y", "z"]:
        cols = find_axis_columns(df, axis)
        if len(cols) == 0:
            raise RuntimeError(f"[{csv_path}] 축 '{axis}' 컬럼을 찾지 못함")
        X = df[cols].to_numpy(dtype=np.float64)  # (T, N)
        # NaN 많은 컬럼 제거
        nan_ratio = np.isnan(X).mean(axis=0)
        keep = nan_ratio <= float(drop_nan_ratio)
        if keep.sum() == 0:
            raise RuntimeError(f"[{csv_path}] 축 '{axis}' 사용 가능한 컬럼 없음 (drop_nan_ratio={drop_nan_ratio})")
        X = X[:, keep]
        X = interpolate_and_impute(X)
        s = pca_first_component(X)
        s = maybe_downsample_series(s, max_len=max_len)
        M = mtf_from_series(s, Q=Q)
        Mr = resize_to(M, size=size)
        channels.append(Mr.astype(np.float32))
    out = np.stack(channels, axis=0)  # (3, size, size)
    return out

# =========================
# 1) 배치 실행
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="./DROZY/gaeMuRan/split_eye_landmark_60s/*/*/*.csv",
                    help="입력 CSV 글롭 패턴")
    ap.add_argument("--bins", type=int, default=32, help="MTF 양자화 bin 수(Q)")
    ap.add_argument("--size", type=int, default=112, help="최종 이미지 한 변 크기")
    ap.add_argument("--drop-nan-ratio", type=float, default=0.5, help="NaN 비율 초과 컬럼 드롭 임계값")
    ap.add_argument("--max-len", type=int, default=None, help="MTF 전 시계열 최대 길이(다운샘플, None이면 전체)")
    ap.add_argument("--src-seg", type=str, default="split_eye_landmark_60s", help="치환될 원본 경로 세그먼트명")
    ap.add_argument("--dst-seg", type=str, default="split_eye_landmark_npy_60s", help="치환 후 경로 세그먼트명")
    ap.add_argument("--overwrite", action="store_true", help="이미 존재해도 덮어쓰기")
    args = ap.parse_args()

    csv_files = glob.glob(args.glob)
    print(f"대상 파일 수: {len(csv_files)}")

    ok, fail = 0, 0
    for i, f in enumerate(csv_files, 1):
        csv_path = Path(f)
        out_path = map_out_path(csv_path, src_segment=args.src_seg, dst_segment=args.dst_seg)
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(csv_files)}] SKIP 존재함: {out_path}")
            ok += 1
            continue
        try:
            arr = build_3ch_mtf_from_csv(
                csv_path=csv_path,
                Q=args.bins,
                size=args.size,
                drop_nan_ratio=args.drop_nan_ratio,
                max_len=args.max_len,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, arr.astype(np.float32))
            print(f"[{i}/{len(csv_files)}] OK → {out_path} | shape={arr.shape}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(csv_files)}] FAIL {csv_path} :: {e}")
            fail += 1

    print(f"완료: 성공 {ok}, 실패 {fail}")

if __name__ == "__main__":
    main()
