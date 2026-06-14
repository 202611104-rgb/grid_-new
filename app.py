# -*- coding: utf-8 -*-
"""
스마트 그리드 안정성 예측 앱 (Streamlit)  — 디자인 리뉴얼 버전
- 노트북에서 만든 SVM 파이프라인(표준화 + SVC rbf)을 그대로 재현합니다.
- 실행: 터미널에서  streamlit run app.py

[핵심 로직 — 원본 그대로 유지]
 1) SVC 에 random_state=42 고정 -> 확률 보정이 실행마다 뒤집히지 않음
 2) 화면 라벨을 '확률(P_stable>=0.5)'에서 직접 도출 -> 라벨과 % 가 절대 어긋나지 않음
 3) 앱이 만든 전용 캐시 파일(grid_svm_model.pkl / grid_scaler.pkl)만 불러옴
    -> 예전에 만들어진 시드 없는 pkl 을 자동으로 무시하고 CSV 로 새로 학습

[이번 변경] 화면(UI) 디자인만 전력 제어실 대시보드 테마로 개선했습니다.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
import streamlit as st

FEATURES = ['tau1', 'tau2', 'tau3', 'tau4',
            'p1', 'p2', 'p3', 'p4',
            'g1', 'g2', 'g3', 'g4']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = []
for d in [BASE_DIR, os.getcwd()]:
    if d and d not in SEARCH_DIRS:
        SEARCH_DIRS.append(d)

# 앱 전용 캐시 파일 (이 이름으로만 저장/로드 -> 예전 pkl 과 섞이지 않음)
MODEL_CACHE = os.path.join(BASE_DIR, "grid_svm_model.pkl")
SCALER_CACHE = os.path.join(BASE_DIR, "grid_scaler.pkl")


def is_grid_csv(path):
    try:
        cols = pd.read_csv(path, nrows=0).columns
        return set(FEATURES).issubset(set(cols))
    except Exception:
        return False


def find_csv():
    candidates = []
    for d in SEARCH_DIRS:
        candidates += sorted(glob.glob(os.path.join(d, "*augmented*.csv")))
        candidates += sorted(glob.glob(os.path.join(d, "*stability*.csv")))
        candidates += sorted(glob.glob(os.path.join(d, "*.csv")))
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if is_grid_csv(c):
            return c
    return None


def list_found_files():
    lines = []
    for d in SEARCH_DIRS:
        files = sorted(glob.glob(os.path.join(d, "*.csv"))) + sorted(glob.glob(os.path.join(d, "*.pkl")))
        names = [os.path.basename(f) for f in files]
        lines.append(f"📁 {d}\n   " + (", ".join(names) if names else "(csv/pkl 없음)"))
    return "\n".join(lines)


@st.cache_resource
def load_model_and_scaler():
    # 1순위: 앱이 직접 만든 캐시(grid_*.pkl)만 불러옴
    if os.path.exists(MODEL_CACHE) and os.path.exists(SCALER_CACHE):
        return joblib.load(MODEL_CACHE), joblib.load(SCALER_CACHE), "loaded", "grid_svm_model.pkl"

    # 2순위: CSV로 노트북과 동일하게 학습 (시드 고정)
    csv_path = find_csv()
    if csv_path:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        df = pd.read_csv(csv_path)
        if 'stabf' in df.columns:
            df = df.rename(columns={'stabf': 'stability'})
        if df['stability'].dtype == object:
            y = df['stability'].map({'stable': 1, 'unstable': 0})
        else:
            y = df['stability']
        X = df[FEATURES]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = SVC(kernel='rbf', C=1, probability=True, random_state=42)   # 시드 고정!
        model.fit(X_train_scaled, y_train)

        try:
            joblib.dump(model, MODEL_CACHE)
            joblib.dump(scaler, SCALER_CACHE)
        except Exception:
            pass
        return model, scaler, "trained", os.path.basename(csv_path)

    return None, None, "missing", None


# ===========================================================================
# 화면 (UI) — 전력 제어실 대시보드 테마
# ===========================================================================
st.set_page_config(page_title="스마트 그리드 안정성 예측", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# 디자인: 폰트 + 격자 배경 + 컴포넌트 스타일
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Orbitron:wght@600;700;900&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root{
  --bg:#0A1628;
  --panel:#0F1E36;
  --panel-2:#13243F;
  --line:rgba(56,189,248,0.16);
  --cyan:#22D3EE;
  --cyan-soft:#38BDF8;
  --stable:#34D399;
  --unstable:#FB7185;
  --amber:#FBBF24;
  --text:#E6EEF8;
  --muted:#8AA0BE;
}

/* ----- 전체 배경: 전력망 격자 + 상단 글로우 ----- */
.stApp{
  background-color: var(--bg);
  background-image:
    linear-gradient(rgba(56,189,248,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56,189,248,0.05) 1px, transparent 1px),
    radial-gradient(900px 420px at 50% -8%, rgba(34,211,238,0.16), transparent 70%);
  background-size: 44px 44px, 44px 44px, 100% 100%;
}
.block-container{ max-width: 1080px; padding-top: 1.6rem; padding-bottom: 4rem; }

html, body, [class*="css"]{ font-family:'Pretendard', sans-serif; color: var(--text); }

/* ----- 히어로 헤더 ----- */
.grid-hero{
  position: relative;
  border:1px solid var(--line);
  border-radius:20px;
  padding:30px 34px 26px;
  margin-bottom:22px;
  background:
    linear-gradient(180deg, rgba(19,36,63,0.92), rgba(10,22,40,0.78));
  box-shadow: 0 0 0 1px rgba(34,211,238,0.06) inset, 0 24px 60px -28px rgba(0,0,0,0.8);
  overflow:hidden;
}
.grid-hero::after{
  content:""; position:absolute; right:-60px; top:-60px;
  width:240px; height:240px; border-radius:50%;
  background: radial-gradient(circle, rgba(34,211,238,0.22), transparent 70%);
  filter: blur(6px);
}
.hero-eyebrow{
  font-family:'JetBrains Mono', monospace; font-size:12px; letter-spacing:.28em;
  color:var(--cyan); text-transform:uppercase; display:flex; align-items:center; gap:10px;
}
.hero-dot{ width:8px; height:8px; border-radius:50%; background:var(--stable);
  box-shadow:0 0 0 0 rgba(52,211,153,.6); animation:pulse 1.8s infinite; }
@keyframes pulse{
  0%{ box-shadow:0 0 0 0 rgba(52,211,153,.55); }
  70%{ box-shadow:0 0 0 10px rgba(52,211,153,0); }
  100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
}
.grid-hero h1{
  font-family:'Pretendard', sans-serif; font-weight:800;
  font-size:34px; line-height:1.15; margin:12px 0 8px; letter-spacing:-0.5px;
}
.grid-hero h1 .zap{
  font-family:'Orbitron', sans-serif; color:var(--cyan);
  text-shadow:0 0 18px rgba(34,211,238,.6);
}
.grid-hero p{ color:var(--muted); font-size:15px; margin:0; max-width:640px; }

/* 전류가 흐르는 라인 */
.powerline{ position:relative; height:3px; margin-top:20px; border-radius:3px;
  background:rgba(56,189,248,0.12); overflow:hidden; }
.powerline::before{
  content:""; position:absolute; top:0; left:-40%; height:100%; width:40%;
  background:linear-gradient(90deg, transparent, var(--cyan), #A5F3FC, var(--cyan), transparent);
  box-shadow:0 0 14px var(--cyan);
  animation:flow 2.6s linear infinite;
}
@keyframes flow{ 0%{ left:-45%; } 100%{ left:105%; } }

/* ----- 섹션 헤더 ----- */
.sec-label{
  font-family:'JetBrains Mono', monospace; font-size:12px; letter-spacing:.22em;
  text-transform:uppercase; color:var(--cyan-soft); margin:6px 0 14px;
  display:flex; align-items:center; gap:10px;
}
.sec-label::before{ content:""; width:18px; height:2px; background:var(--cyan); display:inline-block; }

/* ----- 입력 그룹 카드 ----- */
.param-card{
  border:1px solid var(--line); border-radius:16px; padding:16px 18px 6px;
  background:linear-gradient(180deg, rgba(15,30,54,0.9), rgba(10,22,40,0.6));
  height:100%;
}
.param-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.param-ico{ width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  font-size:18px; background:rgba(34,211,238,0.1); border:1px solid var(--line); }
.param-title{ font-weight:700; font-size:15px; }
.param-sub{ font-family:'JetBrains Mono', monospace; font-size:11px; color:var(--muted);
  letter-spacing:.05em; }

/* ----- 슬라이더 ----- */
[data-testid="stSlider"] label{ color:var(--muted)!important; font-family:'JetBrains Mono', monospace;
  font-size:12px!important; letter-spacing:.04em; }
[data-baseweb="slider"] div[role="slider"]{ box-shadow:0 0 10px rgba(34,211,238,.6); }

/* ----- 버튼 ----- */
.stButton > button{
  font-family:'JetBrains Mono', monospace; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; border:none; border-radius:14px; padding:14px 0; font-size:15px;
  color:#021018;
  background:linear-gradient(135deg, #22D3EE, #38BDF8 60%, #818CF8);
  box-shadow:0 14px 34px -14px rgba(34,211,238,.8), 0 0 0 1px rgba(34,211,238,.4) inset;
  transition:transform .12s ease, box-shadow .2s ease;
}
.stButton > button:hover{ transform:translateY(-2px);
  box-shadow:0 20px 44px -14px rgba(34,211,238,1), 0 0 0 1px rgba(165,243,252,.6) inset; }
.stButton > button:active{ transform:translateY(0); }

/* ----- 결과 카드 ----- */
.result-card{
  border-radius:20px; padding:26px 28px; margin-top:8px; position:relative; overflow:hidden;
  border:1px solid var(--line);
}
.result-stable{ background:linear-gradient(135deg, rgba(16,185,129,0.16), rgba(10,22,40,0.6));
  border-color:rgba(52,211,153,0.4); }
.result-unstable{ background:linear-gradient(135deg, rgba(244,63,94,0.16), rgba(10,22,40,0.6));
  border-color:rgba(251,113,133,0.4); }
.result-flag{ font-family:'JetBrains Mono', monospace; font-size:12px; letter-spacing:.2em;
  text-transform:uppercase; color:var(--muted); }
.result-title{ font-family:'Orbitron', sans-serif; font-weight:900; font-size:30px; margin:6px 0 2px;
  display:flex; align-items:center; gap:12px; }
.result-stable .result-title{ color:var(--stable); text-shadow:0 0 22px rgba(52,211,153,.45); }
.result-unstable .result-title{ color:var(--unstable); text-shadow:0 0 22px rgba(251,113,133,.45); }
.result-sub{ color:var(--muted); font-size:14px; }

/* 확률 게이지 */
.gauge-wrap{ margin-top:20px; }
.gauge-top{ display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace;
  font-size:12px; color:var(--muted); letter-spacing:.08em; margin-bottom:8px; }
.gauge-bar{ height:14px; border-radius:10px; background:rgba(255,255,255,0.06);
  overflow:hidden; border:1px solid var(--line); }
.gauge-fill{ height:100%; border-radius:10px;
  background:linear-gradient(90deg, var(--unstable) 0%, var(--amber) 50%, var(--stable) 100%);
  box-shadow:0 0 16px rgba(52,211,153,.4); transition:width .6s ease; }
.gauge-readout{ display:flex; gap:14px; margin-top:18px; }
.readout{ flex:1; border:1px solid var(--line); border-radius:14px; padding:14px 16px;
  background:rgba(10,22,40,0.55); }
.readout .k{ font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); }
.readout .v{ font-family:'JetBrains Mono', monospace; font-weight:800; font-size:26px; margin-top:4px; }
.readout.s .v{ color:var(--stable); }
.readout.u .v{ color:var(--unstable); }

/* ----- 상태 배너 ----- */
.status-line{ font-family:'JetBrains Mono', monospace; font-size:12.5px; letter-spacing:.04em;
  border:1px solid var(--line); border-radius:12px; padding:10px 14px; margin-bottom:18px;
  display:flex; align-items:center; gap:10px; color:var(--muted);
  background:rgba(15,30,54,0.6); }
.status-line .ok{ color:var(--stable); } .status-line .info{ color:var(--cyan-soft); }

/* 푸터 캡션 */
.foot{ font-family:'JetBrains Mono', monospace; font-size:11.5px; color:var(--muted);
  letter-spacing:.05em; text-align:center; margin-top:30px; }

[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px;
  background:rgba(10,22,40,0.5); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.markdown("""
<div class="grid-hero">
  <div class="hero-eyebrow"><span class="hero-dot"></span> SMART GRID · STABILITY MONITOR</div>
  <h1><span class="zap">⚡</span> 스마트 그리드 안정성 예측</h1>
  <p>12개의 하드웨어 파라미터(응답시간 τ · 전력 p · 가격탄력성 g)를 입력하면,
     SVM 모델이 전력망의 <b>안정 / 불안정</b> 상태를 실시간으로 판정합니다.</p>
  <div class="powerline"></div>
</div>
""", unsafe_allow_html=True)

model, scaler, status, source = load_model_and_scaler()

if status == "missing":
    st.error("데이터(CSV)를 찾지 못했습니다. 아래 폴더 중 한 곳에 "
             "`smart_grid_stability_augmented.csv` 를 넣어주세요.")
    st.markdown("**현재 검색한 폴더와 그 안의 파일들:**")
    st.code(list_found_files())
    st.stop()

if status == "trained":
    st.markdown(f'<div class="status-line"><span class="info">●</span> '
                f'CSV(<b>{source}</b>)로 학습 완료 — 다음 실행부터는 캐시로 빠르게 로드됩니다.</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status-line"><span class="ok">●</span> '
                'ONLINE — 학습된 모델 캐시를 불러왔습니다. 입력 준비 완료.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
st.markdown('<div class="sec-label">Input Parameters · 입력값</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="param-head">
      <div class="param-ico">⏱️</div>
      <div><div class="param-title">응답 시간</div><div class="param-sub">tau · τ1 – τ4</div></div>
    </div>""", unsafe_allow_html=True)
    tau1 = st.slider("tau1", 0.5, 10.0, 5.25, 0.1)
    tau2 = st.slider("tau2", 0.5, 10.0, 5.25, 0.1)
    tau3 = st.slider("tau3", 0.5, 10.0, 5.25, 0.1)
    tau4 = st.slider("tau4", 0.5, 10.0, 5.25, 0.1)

with col2:
    st.markdown("""
    <div class="param-head">
      <div class="param-ico">🔌</div>
      <div><div class="param-title">전력 생산/소비</div><div class="param-sub">power · p1 – p4</div></div>
    </div>""", unsafe_allow_html=True)
    p1 = st.slider("p1 (생산)", 1.5, 6.0, 3.75, 0.1)
    p2 = st.slider("p2 (소비)", -2.0, -0.5, -1.25, 0.05)
    p3 = st.slider("p3 (소비)", -2.0, -0.5, -1.25, 0.05)
    p4 = st.slider("p4 (소비)", -2.0, -0.5, -1.25, 0.05)

with col3:
    st.markdown("""
    <div class="param-head">
      <div class="param-ico">📈</div>
      <div><div class="param-title">가격 탄력성</div><div class="param-sub">gamma · g1 – g4</div></div>
    </div>""", unsafe_allow_html=True)
    g1 = st.slider("g1", 0.05, 1.0, 0.525, 0.01)
    g2 = st.slider("g2", 0.05, 1.0, 0.525, 0.01)
    g3 = st.slider("g3", 0.05, 1.0, 0.525, 0.01)
    g4 = st.slider("g4", 0.05, 1.0, 0.525, 0.01)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

if st.button("⚡ 안정성 예측 실행", type="primary", use_container_width=True):
    x = pd.DataFrame(
        [[tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]],
        columns=FEATURES
    )
    x_scaled = scaler.transform(x)

    prob_stable = float(model.predict_proba(x_scaled)[0][1])   # 안정(stable)일 확률
    pred = 1 if prob_stable >= 0.5 else 0                       # 라벨을 확률에서 직접 도출(일관성 보장)

    pct_stable = prob_stable * 100
    pct_unstable = (1 - prob_stable) * 100

    if pred == 1:
        card_cls, flag, icon, title, sub = (
            "result-stable", "GRID STATUS · 판정 결과", "🟢", "안정 STABLE",
            "입력된 운전 조건에서 전력망이 안정적으로 동작할 것으로 예측됩니다."
        )
    else:
        card_cls, flag, icon, title, sub = (
            "result-unstable", "GRID STATUS · 판정 결과", "🔴", "불안정 UNSTABLE",
            "입력된 운전 조건에서 전력망이 불안정해질 위험이 있습니다."
        )

    st.markdown(f"""
    <div class="result-card {card_cls}">
      <div class="result-flag">{flag}</div>
      <div class="result-title">{icon} {title}</div>
      <div class="result-sub">{sub}</div>

      <div class="gauge-wrap">
        <div class="gauge-top"><span>UNSTABLE</span><span>P(stable) = {pct_stable:.1f}%</span><span>STABLE</span></div>
        <div class="gauge-bar"><div class="gauge-fill" style="width:{pct_stable:.1f}%"></div></div>
        <div class="gauge-readout">
          <div class="readout s"><div class="k">안정 확률</div><div class="v">{pct_stable:.1f}%</div></div>
          <div class="readout u"><div class="k">불안정 확률</div><div class="v">{pct_unstable:.1f}%</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("입력값 다시 보기"):
        st.dataframe(x, use_container_width=True)

st.markdown('<div class="foot">※ 안정=1, 불안정=0 으로 인코딩된 노트북 SVM(rbf) 모델 기준 · seed 42 고정</div>',
            unsafe_allow_html=True)