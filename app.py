from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import dotenv_values
from hashlib import md5
import hashlib
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import qdrant_client.models as models

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"


env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']

def get_api_key_securely():
    if "api_key_verified" not in st.session_state:
        st.session_state["api_key_verified"] = False
    if not st.session_state["api_key_verified"]:
        api_key = st.text_input(
            "Wklej swój klucz API OpenAI ",
            type="password",
            placeholder="sk-…",
            key="raw_api_key"
        )
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                client.models.list()
                user_id = hashlib.md5(api_key.encode('utf-8')).hexdigest()
                st.session_state["openai_api_key"] = api_key
                st.session_state["user_id"] = user_id
                st.session_state["api_key_verified"] = True
                st.rerun()
            except Exception:
                st.error("Nieprawidłowy klucz API")
        st.stop()
    return st.session_state["openai_api_key"]

api_key = get_api_key_securely()
openai_client = OpenAI(api_key=api_key)

def transcribe_audio(audio_bytes):
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )
    return transcript.text

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"])

def assure_db_collection_exists():
    client = get_qdrant_client()
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="user_id",
            field_schema="keyword"
        )
    except:
        pass

def get_embeddings(text):
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

def add_note_to_db(note_text):
    client = get_qdrant_client()
    user_id = st.session_state["user_id"]
    count = client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[PointStruct(
            id=count + 1,
            vector=get_embeddings(note_text),
            payload={"text": note_text, "user_id": user_id},
        )],
    )
    if st.session_state.get("has_searched"):
        st.session_state["search_results"] = list_notes_from_db(st.session_state["last_query"])

def list_notes_from_db(query=None):
    client = get_qdrant_client()
    user_id = st.session_state["user_id"]
    flt = models.Filter(must=[
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ])
    if query and query.strip():
        notes = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embeddings(query),
            query_filter=flt,
            limit=10,
        )
    else:
        notes = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=flt,
            limit=10
        )[0]
    return [{"text": n.payload["text"], "score": getattr(n, "score", None)} for n in notes]

# MAIN
st.set_page_config(page_title="Audio Notatki", layout="centered")
assure_db_collection_exists()
st.title("Audio Notatki")

# Track active tab
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0
# Init search state
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "has_searched" not in st.session_state:
    st.session_state["has_searched"] = False

tabs = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])
add_tab, search_tab = tabs

with add_tab:
    st.session_state["active_tab"] = 0
    note_audio = audiorecorder(start_prompt="Nagraj notatkę", stop_prompt="Zatrzymaj nagrywanie")
    if note_audio:
        buf = BytesIO(); note_audio.export(buf, format="mp3")
        st.session_state["note_audio_bytes"] = buf.getvalue()
        md = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state.get("note_audio_bytes_md5") != md:
            st.session_state.update(
                note_audio_bytes_md5=md,
                note_audio_text="",
                note_text=""
            )
        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")
        if st.button("Transkrybuj audio"):
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])
        if st.session_state.get("note_audio_text"):
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])
        if st.session_state.get("note_text") and st.button("Zapisz notatkę"):
            add_note_to_db(st.session_state["note_text"])
            st.toast("Notatka zapisana", icon="🎉")

with search_tab:
    st.session_state["active_tab"] = 1
    q = st.text_input("Wyszukaj notatkę", value=st.session_state["last_query"])
    if st.button("Szukaj"):
        st.session_state["last_query"] = q
        st.session_state["search_results"] = list_notes_from_db(q)
        st.session_state["has_searched"] = True
        st.rerun()
    
    if st.session_state["search_results"]:
        cnt = len(st.session_state["search_results"])
        hdr = (f"Wyniki wyszukiwania ({cnt})"
               if st.session_state["last_query"].strip()
               else f"Wszystkie notatki ({cnt})")
        st.subheader(hdr)
        for note in st.session_state["search_results"]:
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"]:
                    st.markdown(f':violet[{note["score"]}]')
    elif st.session_state["has_searched"]:
        st.info("Brak wyników dla tego zapytania")
