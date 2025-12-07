import streamlit as st
import pandas as pd
import time
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

st.sidebar.info("Оценка производится по 8 критериям:\n\n1. Безвредность\n2. Достоверность\n3. Полезность\n4. Полнота\n5. Лаконичность\n6. Актуальность\n7. Уместность\n8. Читаемость")

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
            with st.spinner("Запрос к YandexGPT..."):
                result = evaluate_with_yandex(
                    query=user_query,
                    ans_a=ans_a,
                    ans_b=ans_b,
                    api_key=api_key,
                    folder_id=folder_id,
                    demo_mode=demo_mode
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
                            demo_mode=demo_mode
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
                            
                        results.append(row_result)
                        progress_bar.progress((index + 1) / total_rows)
                    
                    status_text.text("Готово!")
                    progress_bar.empty()
                    
                    # Result Dataframe
                    result_df = pd.DataFrame(results)
                    
                    st.success("Пакетная обработка завершена!")
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

