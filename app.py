import streamlit as st
import pandas as pd
import time
import altair as alt
from judge_logic import evaluate_with_yandex

# Page Config
st.set_page_config(
    page_title="АвтоАсессор: YandexGPT-as-a-Judge",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for nicer UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #fc3f1d; /* Yandex Red/Orange-ish */
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #fc3f1d;
    }
</style>
""", unsafe_allow_html=True)


# 1. Sidebar (Global Settings)
st.sidebar.title("⚙️ Настройки")

demo_mode = st.sidebar.checkbox("Демо-режим (Mock)", value=True)

api_key = st.sidebar.text_input("Yandex IAM Token / API Key", type="password", disabled=demo_mode)
folder_id = st.sidebar.text_input("Yandex Folder ID", disabled=demo_mode)
if not demo_mode:
    st.sidebar.caption("Folder ID is required to access YandexGPT resources in your cloud.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Персона Судьи")
persona_name = st.sidebar.selectbox(
    "Выберите стиль оценки",
    ["Strict Fact-Checker", "Helpful Editor"]
)

if persona_name == "Strict Fact-Checker":
    st.sidebar.info("🧐 **Strict Fact-Checker**: Фокус на точности фактов. Жестко штрафует за галлюцинации.")
else:
    st.sidebar.info("✍️ **Helpful Editor**: Фокус на стиле, структуре и тоне. Ценит форматирование.")

st.sidebar.markdown("---")
st.sidebar.info("Оценка производится по 8 критериям:\n\n1. Безвредность\n2. Достоверность\n3. Полезность\n4. Полнота\n5. Лаконичность\n6. Актуальность\n7. Уместность\n8. Читаемость")

# Analytics Function
def show_analytics(df):
    st.markdown("### 📊 Аналитика и Токеномика")
    
    # 1. Prepare Data
    total = len(df)
    
    # Derive winner column if not present
    if "winner" not in df.columns:
        def get_winner(row):
            sa = row.get("score_a_overall", 0)
            sb = row.get("score_b_overall", 0)
            if sa > sb: return "Model A"
            elif sb > sa: return "Model B"
            else: return "Tie"
        df["winner"] = df.apply(get_winner, axis=1)

    # 2. Key Metrics (Quality)
    model_a_wins = len(df[df["winner"] == "Model A"])
    model_b_wins = len(df[df["winner"] == "Model B"])
    
    win_rate_a = (model_a_wins / total * 100) if total > 0 else 0
    win_rate_b = (model_b_wins / total * 100) if total > 0 else 0
    
    avg_score_a = df["score_a_overall"].mean()
    avg_score_b = df["score_b_overall"].mean()
    
    st.markdown("#### 🏆 Качество Моделей")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win Rate (Model A)", f"{win_rate_a:.1f}%")
    m2.metric("Win Rate (Model B)", f"{win_rate_b:.1f}%")
    m3.metric("Ср. балл (Model A)", f"{avg_score_a:.1f}")
    m4.metric("Ср. балл (Model B)", f"{avg_score_b:.1f}")
    
    # 3. Tokenomics
    st.markdown("#### 💰 Токеномика")
    
    # Ensure token columns exist and are numeric
    for col in ["input_tokens", "output_tokens", "total_tokens"]:
        if col not in df.columns:
            df[col] = 0
            
    total_input = df["input_tokens"].sum()
    total_output = df["output_tokens"].sum()
    grand_total_tokens = df["total_tokens"].sum()
    
    # Cost Estimation (Mock: 0.40 RUB per 1k tokens combined for simplicity)
    est_cost = (grand_total_tokens / 1000) * 0.40
    
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Tokens", f"{grand_total_tokens:,}")
    t2.metric("Avg Tokens / Query", f"{int(grand_total_tokens / total) if total else 0}")
    t3.metric("Est. Cost (₽)", f"₽{est_cost:.2f}", help="Расчетная стоимость: 0.40 ₽ за 1k токенов")
    
    # 4. Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.caption("Распределение побед")
        winner_counts = df["winner"].value_counts().reset_index()
        winner_counts.columns = ["Winner", "Count"]

        chart = alt.Chart(winner_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Winner", type="nominal", scale=alt.Scale(domain=['Model A', 'Model B', 'Tie'], range=['#fc3f1d', '#4a90e2', '#999999'])),
            tooltip=["Winner", "Count"]
        )
        st.altair_chart(chart, use_container_width=True)
        
    with c2:
        st.caption("Использование токенов по запросам")
        # Prepare data for stacked bar chart: Query Index -> Input, Output
        # Altair needs long format for stacked bars
        
        # Add index column for X-axis
        df_chart = df.reset_index()
        
        token_chart_data = df_chart.melt(id_vars=["index"], value_vars=["input_tokens", "output_tokens"], var_name="Token Type", value_name="Count")
        
        bar_chart = alt.Chart(token_chart_data).mark_bar().encode(
            x=alt.X("index:O", title="Номер запроса"),
            y=alt.Y("Count:Q", title="Количество токенов"),
            color=alt.Color("Token Type", scale=alt.Scale(domain=['input_tokens', 'output_tokens'], range=['#9CA3AF', '#F59E0B'])),
            tooltip=["index", "Token Type", "Count"]
        )
        st.altair_chart(bar_chart, use_container_width=True)


# Main Title
st.title("⚖️ АвтоАсессор: YandexGPT-as-a-Judge")
st.markdown("LLM-as-a-Judge для оценки качества поисковых ответов powered by **YandexGPT**.")

# Tabs
tab_single, tab_batch = st.tabs(["📝 Одиночный режим", "🚀 Пакетная обработка"])

# --- TAB 1: SINGLE MODE ---
with tab_single:
    user_query = st.text_input("Запрос", placeholder="Вставьте запрос...")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ответ Model A")
        ans_a = st.text_area("Ответ A", height=200, placeholder="Вставьте ответ Model A...")

    with col2:
        st.subheader("Ответ Model B")
        ans_b = st.text_area("Ответ B", height=200, placeholder="Вставьте ответ Model B...")

    # Logic
    if st.button("Оценить!", type="primary", key="btn_single"):
        if not user_query or not ans_a or not ans_b:
            st.warning("Пожалуйста, заполните запрос и оба ответа.")
        else:
            with st.spinner(f"Запрос к YandexGPT ({persona_name})..."):
                result = evaluate_with_yandex(
                    query=user_query,
                    ans_a=ans_a,
                    ans_b=ans_b,
                    api_key=api_key,
                    folder_id=folder_id,
                    demo_mode=demo_mode,
                    persona_name=persona_name
                )

            if "error" in result:
                st.error(f"Error: {result['error']}")
                if "raw_response" in result:
                    with st.expander("Показать Raw Response"):
                        st.code(result["raw_response"])
            else:
                # Display Results
                st.markdown("---")
                st.success("Оценка завершена!")
                
                # Comparison Summary
                st.markdown(f"### 💡 Сравнительный вердикт")
                st.info(result.get('comparison', 'Нет сравнительного вердикта.'))
                
                # Show Token Usage for Single Mode too
                if "usage" in result:
                    usage = result["usage"]
                    st.caption(f"💰 Tokens: {usage.get('totalTokens')} (In: {usage.get('inputTextTokens')}, Out: {usage.get('completionTokens')})")

                res_col1, res_col2 = st.columns(2)
                
                # Helper to display model stats
                def display_model_stats(container, model_key, title):
                    data = result.get(model_key, {})
                    container.markdown(f"## {title}")
                    container.metric("Overall Score", f"{data.get('overall_score', 0)}/10")
                    
                    container.markdown("#### Обоснование")
                    container.caption(data.get('reasoning', "Нет объяснения."))
                    
                    container.markdown("#### Критерии")
                    scores = data.get('scores', {})
                    for k, v in scores.items():
                        container.progress(v / 10, text=f"{k}: {v}/10")

                with res_col1:
                    display_model_stats(st, "model_a", "Model A")
                
                with res_col2:
                    display_model_stats(st, "model_b", "Model B")

# --- TAB 2: BATCH MODE ---
with tab_batch:
    st.markdown("### Загрузите CSV файл")
    st.markdown("Файл должен содержать столбцы: `query`, `answer_a`, `answer_b`.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    # Clear state if new file uploaded (optional UX choice, keeping simple for now)
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validation
            required_cols = {'query', 'answer_a', 'answer_b'}
            if not required_cols.issubset(df.columns):
                st.error(f"Ошибка: В CSV файле отсутствуют обязательные столбцы: {required_cols - set(df.columns)}")
            else:
                st.markdown("#### Предпросмотр (первые 5 строк):")
                st.dataframe(df.head())
                
                if st.button("Начать пакетную оценку", type="primary", key="btn_batch"):
                    
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_rows = len(df)
                    
                    for index, row in df.iterrows():
                        status_text.text(f"Обработка строки {index + 1} из {total_rows}...")
                        
                        # Demo mode throttling
                        if demo_mode:
                            time.sleep(0.1)
                        
                        eval_res = evaluate_with_yandex(
                            query=row['query'],
                            ans_a=row['answer_a'],
                            ans_b=row['answer_b'],
                            api_key=api_key,
                            folder_id=folder_id,
                            demo_mode=demo_mode,
                            persona_name=persona_name
                        )
                        
                        # Parse Result for CSV
                        row_result = row.to_dict()
                        if "error" in eval_res:
                            row_result["error"] = eval_res["error"]
                        else:
                            # Model A Stats
                            ma = eval_res.get("model_a", {})
                            row_result["score_a_overall"] = ma.get("overall_score")
                            row_result["reasoning_a"] = ma.get("reasoning")
                            
                            # Model B Stats
                            mb = eval_res.get("model_b", {})
                            row_result["score_b_overall"] = mb.get("overall_score")
                            row_result["reasoning_b"] = mb.get("reasoning")
                            
                            row_result["comparison"] = eval_res.get("comparison")
                            
                            # Token Usage
                            usage = eval_res.get("usage", {})
                            row_result["input_tokens"] = int(usage.get("inputTextTokens", 0))
                            row_result["output_tokens"] = int(usage.get("completionTokens", 0))
                            row_result["total_tokens"] = int(usage.get("totalTokens", 0))
                            
                        results.append(row_result)
                        progress_bar.progress((index + 1) / total_rows)
                    
                    status_text.text("Готово!")
                    progress_bar.empty()
                    
                    # Store in Session State
                    result_df = pd.DataFrame(results)
                    st.session_state['batch_results'] = result_df
                    
                # Display Results from Session State
                if 'batch_results' in st.session_state:
                    result_df = st.session_state['batch_results']
                    
                    with st.expander("📊 Отчет об оценке", expanded=True):
                        st.success("Пакетная обработка завершена!")
                        
                        # Analytics
                        show_analytics(result_df)
                        
                        st.markdown("### Детализация")
                        st.dataframe(result_df)
                        
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать результаты (CSV)",
                            data=csv,
                            file_name="evaluation_results.csv",
                            mime="text/csv",
                        )
                    
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
