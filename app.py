import streamlit as st
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


# 1. Sidebar
st.sidebar.title("⚙️ Настройки")

demo_mode = st.sidebar.checkbox("Демо-режим (Mock)", value=True)

api_key = st.sidebar.text_input("Yandex IAM Token / API Key", type="password", disabled=demo_mode)
folder_id = st.sidebar.text_input("Yandex Folder ID", disabled=demo_mode)
if not demo_mode:
    st.sidebar.caption("Folder ID is required to access YandexGPT resources in your cloud.")

# Criteria selector removed as it's now fixed to 8 criteria.
st.sidebar.info("Оценка производится по 8 критериям:\n\n1. Безвредность\n2. Достоверность\n3. Полезность\n4. Полнота\n5. Лаконичность\n6. Актуальность\n7. Уместность\n8. Читаемость")

# 2. Main Screen
st.title("⚖️ АвтоАсессор: YandexGPT-as-a-Judge")
st.markdown("LLM-as-a-Judge для оценки качества поисковых ответов powered by **YandexGPT**.")

user_query = st.text_input("Запрос", placeholder="Вставьте запрос...")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ответ Model A")
    ans_a = st.text_area("Ответ A", height=200, placeholder="Вставьте ответ Model A...")

with col2:
    st.subheader("Ответ Model B")
    ans_b = st.text_area("Ответ B", height=200, placeholder="Вставьте ответ Model B...")

# 3. Logic
if st.button("Оценить!", type="primary"):
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

