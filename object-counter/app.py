import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("🔢 Object Counting Web App")
st.write("이미지를 업로드하면 노이즈 제거 → 임계값 처리 → Connected Component 분석으로 객체 수를 계산합니다.")


def otsu_threshold(img):
    """Otsu 방법으로 자동 임계값 계산 (numpy만 사용)"""
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, best_var, best_t = 0.0, 0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ 파라미터 설정")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif"])
ksize = st.sidebar.slider("Median filter 커널 크기 (홀수)", 1, 25, 7, step=2)
use_otsu = st.sidebar.checkbox("Otsu 자동 임계값 사용", value=False)
manual_th = st.sidebar.slider("임계값 (수동)", 0, 255, 128, disabled=use_otsu)
invert = st.sidebar.checkbox("마스크 반전 (객체가 배경보다 어두운 경우)", value=False)
min_size = st.sidebar.number_input("최소 객체 크기 (pixel) - 이보다 작으면 노이즈로 간주",
                                   min_value=0, value=1000, step=100)

# ---------------- Main ----------------
if uploaded is None:
    st.info("왼쪽 사이드바에서 이미지를 업로드해주세요.")
    st.stop()

# 1. grayscale 로드
img = Image.open(uploaded).convert("L")
f = np.array(img)

# 2. 노이즈 제거
f_denoised = signal.medfilt2d(f, ksize)

# 3. thresholding
th_val = otsu_threshold(f_denoised) if use_otsu else manual_th
f_mask = f_denoised > th_val
if invert:
    f_mask = ~f_mask

# 4. connected component labeling
labels, nums = nd.label(f_mask)
sizes = nd.sum(f_mask, labels, index=np.arange(1, nums + 1)).astype(int)

# min_size 이상인 객체만 유효
valid_idx = np.where(sizes >= min_size)[0] + 1
valid_sizes = sizes[valid_idx - 1]
valid_mask = np.isin(labels, valid_idx)
valid_labels, valid_n = nd.label(valid_mask)

# ---------------- 결과 출력 ----------------
c1, c2, c3 = st.columns(3)
c1.metric("사용된 임계값", int(th_val))
c2.metric("전체 connected component 수", int(nums))
c3.metric("✅ 최종 객체 수 (min_size 적용)", int(len(valid_idx)))

st.subheader("처리 단계별 결과")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image(f, caption="1. Grayscale 입력", clamp=True, use_container_width=True)
with col2:
    st.image(f_denoised, caption=f"2. Median filter (k={ksize})", clamp=True, use_container_width=True)
with col3:
    st.image((f_mask * 255).astype(np.uint8), caption=f"3. Binary mask (th={th_val})", use_container_width=True)
with col4:
    fig, ax = plt.subplots()
    ax.imshow(np.ma.masked_where(valid_labels == 0, valid_labels), cmap="Spectral_r")
    ax.axis("off")
    ax.set_title(f"4. Labeled objects ({valid_n})")
    st.pyplot(fig)

# 히스토그램
st.subheader("히스토그램")
fig_h, ax_h = plt.subplots(figsize=(8, 2.5))
ax_h.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
ax_h.axvline(th_val, color="red", linestyle="--", label=f"threshold={th_val}")
ax_h.set_xlabel("Pixel Intensity")
ax_h.set_ylabel("Frequency")
ax_h.legend()
st.pyplot(fig_h)

# 객체별 pixel 수 표
st.subheader("객체별 픽셀 수")
if len(valid_idx) == 0:
    st.warning("조건에 맞는 객체가 없습니다. 임계값 또는 최소 객체 크기를 조정해보세요.")
else:
    df = pd.DataFrame({"label": valid_idx, "pixels": valid_sizes}).sort_values("pixels", ascending=False)
    st.dataframe(df, use_container_width=True)

    # 가장 큰 / 가장 작은 객체
    i_max = valid_idx[np.argmax(valid_sizes)]
    i_min = valid_idx[np.argmin(valid_sizes)]

    st.subheader("가장 큰 / 가장 작은 객체")
    fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(f, cmap="gray"); axs[0].set_title("input image")
    axs[1].imshow(labels == i_max, cmap="Spectral_r")
    axs[1].set_title(f"largest (label {i_max}, {sizes[i_max-1]} px)")
    axs[2].imshow(labels == i_min, cmap="Spectral_r")
    axs[2].set_title(f"smallest (label {i_min}, {sizes[i_min-1]} px)")
    for a in axs:
        a.axis("off")
    st.pyplot(fig2)
