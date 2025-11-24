import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 리눅스(Streamlit Cloud) 환경 한글 폰트 설정
if platform.system() == 'Linux':
    plt.rc('font', family='NanumGothic')
else:
    plt.rc('font', family='Malgun Gothic') # 윈도우 로컬 테스트용
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------------------------------------------------------
# 1. 데이터 준비 (Back-end)
# -----------------------------------------------------------------------------
# R 코드의 데이터 구조를 모방한 데모 데이터 생성
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'Production': np.random.normal(100, 15, n),      # 생산량
    'Yield': np.random.uniform(80, 95, n),           # 수율
    'Productivity': np.random.uniform(1.0, 2.5, n),  # 생산성
    'Workforce': np.random.normal(50, 5, n),         # 인원
    'Work_Hour': np.random.normal(180, 10, n),       # 작업시간
    'Defect_Rate': np.random.uniform(0, 5, n)        # 불량률
})

# 변수 간 상관관계 인위적 주입 (시각화 효과를 위해)
df['Production'] = df['Workforce'] * 0.5 + df['Work_Hour'] * 0.3 + np.random.normal(0, 5, n)
df['Productivity'] = df['Production'] / df['Work_Hour']

# -----------------------------------------------------------------------------
# 2. UI 및 시각화 (Front-end)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="상관관계 분석 시스템", layout="wide")

st.title("🧩 생산 지표 상관관계 심층 분석")
st.markdown("""
이 대시보드는 **R의 corrplot 및 PerformanceAnalytics** 기능을 Python으로 구현한 것입니다.  
데이터 간의 연관성을 히트맵, 클러스터링, 산점도 행렬로 분석합니다.
""")
st.divider()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 분석 옵션 설정")
    chart_type = st.selectbox(
        "시각화 스타일 선택",
        ["Numeric Heatmap (숫자)", "Shaded Heatmap (음영)", "Clustermap (군집화)", "Scatter Matrix (분포)"]
    )
    
    st.write("---")
    color_map = st.selectbox("색상 테마 (Colormap)", ["RdBu_r", "coolwarm", "viridis", "Greens"])
    show_values = st.checkbox("상관계수 값 표시", value=True)

# 메인 분석 영역
col1, col2 = st.columns([3, 1])

with col1:
    # 상관계수 행렬 계산
    corr = df.corr()

    # 1. Numeric / Shaded Heatmap (기본 corrplot 유사)
    if chart_type in ["Numeric Heatmap (숫자)", "Shaded Heatmap (음영)"]:
        st.subheader(f"📊 {chart_type}")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 마스크 설정 (대각선 윗부분 가리기 옵션 - R의 diag=FALSE 유사 효과 가능)
        mask = np.triu(np.ones_like(corr, dtype=bool)) if chart_type == "Shaded Heatmap (음영)" else None
        
        sns.heatmap(corr, 
                    annot=show_values, 
                    fmt=".2f", 
                    cmap=color_map, 
                    vmin=-1, vmax=1, 
                    center=0,
                    square=True, 
                    linewidths=.5, 
                    cbar_kws={"shrink": .5},
                    mask=mask,
                    ax=ax)
        
        plt.title("Correlation Matrix", fontsize=15)
        st.pyplot(fig)

    # 2. Clustermap (R의 order="hclust" 유사)
    elif chart_type == "Clustermap (군집화)":
        st.subheader("🧬 Hierarchical Clustering (계층적 군집화)")
        st.info("유사한 패턴을 가진 변수끼리 자동으로 그룹핑하여 보여줍니다. (R의 'hclust' 옵션 기능)")
        
        # seaborn clustermap
        fig = sns.clustermap(corr, 
                             annot=show_values, 
                             fmt=".2f", 
                             cmap=color_map, 
                             vmin=-1, vmax=1, 
                             center=0,
                             figsize=(10, 10),
                             dendrogram_ratio=(.1, .2),
                             cbar_pos=(0, .2, .03, .4))
        st.pyplot(fig)

    # 3. Scatter Matrix (R의 PerformanceAnalytics chart.Correlation 유사)
    elif chart_type == "Scatter Matrix (분포)":
        st.subheader("📈 다변량 분포 및 산점도 (Scatter Matrix)")
        st.info("변수 간의 선형성뿐만 아니라 데이터의 분포(Histogram)와 이상치를 파악합니다.")
        
        fig = sns.pairplot(df, kind="reg", diag_kind="kde", plot_kws={'line_kws':{'color':'red'}})
        st.pyplot(fig)

with col2:
    st.subheader("💡 인사이트 도출")
    st.write("상관계수가 높은 상위 변수 쌍:")
    
    # 상관계수 절대값 기준 정렬 (자기 자신 제외)
    corr_unstacked = corr.abs().unstack()
    corr_sorted = corr_unstacked.sort_values(ascending=False)
    corr_pairs = corr_sorted[corr_sorted < 1].drop_duplicates().head(5)
    
    for idx, value in corr_pairs.items():
        v_real = corr.loc[idx[0], idx[1]] # 실제 부호 확인
        st.markdown(f"**{idx[0]} - {idx[1]}**")
        st.progress(abs(value))
        st.caption(f"Coefficient: {v_real:.2f}")

    st.write("---")
    st.write("**해석 가이드:**")
    st.markdown("- **0.7 이상:** 매우 강한 상관관계")
    st.markdown("- **0.3 ~ 0.7:** 뚜렷한 상관관계")
    st.markdown("- **0.1 미만:** 관계 없음")
