# ui.py

import streamlit as st
from backend import extract_video_id, get_transcript, build_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="YouTube RAG", page_icon="🎥")

st.title("🎥 YouTube ChatGPT")


if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False



with st.sidebar:
    st.header("⚙️ Setup Video")

    youtube_url = st.text_input("Enter YouTube URL")
    language = st.selectbox("Language", ["auto"])

    if st.button("Process Video"):

        video_id = extract_video_id(youtube_url)

        if not video_id:
            st.error("❌ Invalid URL")
        else:
            try:
                with st.spinner("Fetching transcript..."):
                    text = get_transcript(video_id, language)

                with st.spinner("Building AI model..."):
                    chain = build_chain(text)

                st.session_state.chain = chain
                st.session_state.video_processed = True
                st.success("✅ Video ready!")

            except Exception as e:
                st.error(f"❌ {e}")



if not st.session_state.video_processed:
    st.info("👈 Please process a YouTube video from sidebar first")
else:

    
    if youtube_url:
        st.video(youtube_url)

   
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the video...")

    if user_input:

        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        
        with st.chat_message("user"):
            st.markdown(user_input)

        
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    response = st.session_state.chain.invoke(user_input)
                except Exception as e:
                    response = f"❌ Error: {e}"

                st.markdown(response)

        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })