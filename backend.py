# backend.py

import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser



def extract_video_id(url):
    if not url:
        return None

    # handle multiple formats
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:embed/)([0-9A-Za-z_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None



def get_transcript(video_id, language):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        try:
            if language == "auto":
                transcript = list(transcript_list)[0]
            else:
                transcript = transcript_list.find_transcript([language])
        except NoTranscriptFound:
            transcript = list(transcript_list)[0]

        fetched = transcript.fetch()

        
        full_text = " ".join([chunk.text for chunk in fetched])

        return full_text

    except TranscriptsDisabled:
        raise Exception("❌ Captions are disabled")

    except Exception as e:
        raise Exception(f"Transcript Error: {str(e)}")



def build_chain(full_text):

    if not full_text:
        raise Exception("No transcript data available")

    # TEXT SPLITTING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100   # slightly better retrieval
    )
    chunks = splitter.create_documents([full_text])

    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    
    llm = ChatOpenAI()

    
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.
If the context is insufficient, say: "I don't know based on the video."

Context:
{context}

Question: {question}
""",
        input_variables=['context', 'question']
    )

    # FORMAT FUNCTION
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # CHAIN
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain