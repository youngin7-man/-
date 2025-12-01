import streamlit as st
import pandas as pd
import numpy as np
import io

# Streamlit 앱의 기본 설정을 구성합니다.
st.set_page_config(layout="wide", page_title="체력 측정 데이터 상관관계 분석")

# --- 1. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(uploaded_file):
    """CSV 파일을 로드하고 분석에 필요한 전처리를 수행합니다."""
    # 사용자가 업로드한 파일 이름을 기반으로 데이터를 로드합니다.
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    
    # 불필요한 공백이나 특수 문자를 제거하여 컬럼명을 정리합니다.
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(' ', '_')

    # 분석 대상이 될 수 있는 숫자형 데이터만 추출합니다.
    # '신장', '체중', '체지방율', '허리둘레', '악력_좌', '악력_우' 등의 체력 측정 항목을 포함합니다.
    numeric_cols = [
        '나이', '신장', '체중', '체지방율', '허리둘레', 
        '악력_좌', '악력_우', '윗몸말아올리기', '반복점프', '앉아윗몸앞으로굽히기',
        'BMI', '교차윗몸일으키기', '왕복오래달리기', '10M_4회_왕복달리기', '제자리_멀리뛰기',
        '의자에앉았다일어서기', '상대악력', '피부두겹합', '반응시간', '절대악력'
    ]
    
    # 실제 데이터프레임에 존재하는 컬럼만 선택합니다.
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    df_numeric = df[available_numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # 결측치가 너무 많은 행과 열을 제거하여 데이터의 품질을 높입니다.
    df_numeric = df_numeric.dropna(axis=1, how='all') # 모든 값이 NaN인 열 제거
    df_numeric = df_numeric.dropna(how='all') # 모든 값이 NaN인 행 제거
    
    return df_numeric

# --- 2. 상관관계 분석 함수 ---
@st.cache_data
def calculate_correlation(df):
    """상관관계 행렬을 계산하고, 가장 높은 양/음의 상관관계 쌍을 찾습니다."""
    # 상관관계 행렬 계산 (피어슨 상관계수)
    corr_matrix = df.corr()
    
    # 자기 자신과의 상관관계(1.0)를 제외하기 위해 마스크를 적용합니다.
    np.fill_diagonal(corr_matrix.values, np.nan)
    
    # 상관관계를 시리즈 형태로 변환하고 절댓값으로 정렬합니다.
    corr_series = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    # 가장 높은 양의 상관관계 (1에 가장 가까운 값)
    positive_corr = corr_series.iloc[0]
    positive_pair = corr_series.index[0]
    
    # 가장 높은 음의 상관관계 (-1에 가장 가까운 값)
    negative_corr = corr_series.dropna().iloc[-1]
    negative_pair = corr_series.dropna().index[-1]
    
    return positive_pair, positive_corr, negative_pair, negative_corr, corr_matrix

# --- 3. Streamlit 앱 레이아웃 ---
st.title("🏃 체력 측정 데이터 상관관계 분석 앱")
st.markdown("제공된 CSV 파일을 분석하여 다양한 체력 측정 항목 간의 상관관계를 탐색합니다.")

# 파일 로드
df_data = load_data(f'./{st.session_state.uploaded_file}')

if df_data.empty:
    st.error("데이터 로드에 실패했거나, 분석 가능한 숫자형 데이터가 포함되어 있지 않습니다.")
else:
    # 상관관계 분석 실행
    positive_pair, positive_corr, negative_pair, negative_corr, corr_matrix = calculate_correlation(df_data)

    st.header("1. 상관관계 분석 결과 요약")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ 가장 높은 양의 상관관계")
        st.metric(
            label=f"**{positive_pair[0]}** & **{positive_pair[1]}**",
            value=f"{positive_corr:.4f}",
            delta="1에 가장 가까움"
        )
        st.markdown(f"**해석:** 이 두 속성은 함께 증가하거나 감소하는 경향이 **가장 강합니다.**")

    with col2:
        st.subheader("➖ 가장 높은 음의 상관관계")
        st.metric(
            label=f"**{negative_pair[0]}** & **{negative_pair[1]}**",
            value=f"{negative_corr:.4f}",
            delta="-1에 가장 가까움"
        )
        st.markdown(f"**해석:** 이 두 속성은 하나가 증가할 때 다른 하나는 감소하는 경향이 **가장 강합니다.**")

    st.write("---")

    # 4. 버튼 기반 상세 분석 섹션
    st.header("2. 상세 분석 (버튼 클릭)")
    
    # 버튼 레이아웃
    button_col1, button_col2, _ = st.columns([1, 1, 3])

    # 양의 상관관계 버튼
    if button_col1.button("가장 높은 양의 상관관계 보기", type="primary"):
        st.subheader(f"✨ 양의 상관관계 쌍: `{positive_pair[0]}`와 `{positive_pair[1]}`")
        st.success(f"상관 계수: {positive_corr:.4f}")
        
        # 산점도 표시
        st.altair_chart(
            pd.DataFrame({
                'X': df_data[positive_pair[0]],
                'Y': df_data[positive_pair[1]]
            }).corr().reset_index().T,
            use_container_width=True
        )
        st.write("산점도: 두 변수가 우상향으로 밀집되어 있을수록 양의 상관관계가 강합니다.")

    # 음의 상관관계 버튼
    if button_col2.button("가장 높은 음의 상관관계 보기", type="secondary"):
        st.subheader(f"🌪️ 음의 상관관계 쌍: `{negative_pair[0]}`와 `{negative_pair[1]}`")
        st.error(f"상관 계수: {negative_corr:.4f}")

        # 산점도 표시
        st.altair_chart(
            pd.DataFrame({
                'X': df_data[negative_pair[0]],
                'Y': df_data[negative_pair[1]]
            }).corr().reset_index().T,
            use_container_width=True
        )
        st.write("산점도: 두 변수가 우하향으로 밀집되어 있을수록 음의 상관관계가 강합니다.")

    st.write("---")

    # 5. 전체 상관관계 히트맵 (추가 시각화)
    st.header("3. 전체 상관관계 히트맵")
    st.caption("모든 속성 간의 상관관계를 시각적으로 한눈에 확인하세요.")

    # 히트맵 생성
    import altair as alt

    # 상관계수 행렬을 Long Format으로 변환
    corr_df = corr_matrix.stack().reset_index(name='correlation')
    corr_df.columns = ['Variable_1', 'Variable_2', 'Correlation']

    # 자기 자신과의 상관관계 (1.0) 제거
    corr_df = corr_df[corr_df['Variable_1'] != corr_df['Variable_2']]
    
    # 대칭 중복 제거 (Variable_1, Variable_2)와 (Variable_2, Variable_1) 중 하나만 남김
    corr_df['sorted_pair'] = corr_df.apply(lambda row: tuple(sorted((row.Variable_1, row.Variable_2))), axis=1)
    corr_df = corr_df.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])

    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('Variable_1:O', title=None),
        y=alt.Y('Variable_2:O', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(range='diverging'), legend=alt.Legend(title="상관 계수")),
        tooltip=['Variable_1', 'Variable_2', alt.Tooltip('Correlation', format='.4f')]
    ).properties(
        title="체력 측정 항목 간 상관관계 히트맵"
    ).interactive() # 줌 및 팬 활성화

    st.altair_chart(heatmap, use_container_width=True)

# 💡 참고: 파일을 업로드하여 사용자가 직접 선택하는 방식 대신, 
# 사용자님의 요청에 따라 업로드된 파일을 로컬에 존재하는 것으로 가정하고 코드를 작성했습니다.
# 
# ⚠️ 실제 환경에서 실행 시, 'fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv' 파일을
# app.py와 같은 경로에 두시거나, 파일명을 코드의 `st.session_state.uploaded_file`에 맞게 수정해야 합니다.
# 
# **현재 파일명을 세션 상태에 임시 저장하여 처리합니다.**
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = 'fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv'
