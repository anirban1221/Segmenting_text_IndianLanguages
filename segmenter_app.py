import streamlit as st
from PyPDF2 import PdfReader
import docx
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from indicnlp.tokenize.sentence_tokenize import sentence_split
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
# ----- App Title -----
st.image("ChatGPT Image Apr 13, 2025, 07_58_32 PM.png", use_container_width=False,width=400)


# ----- Sidebar Options -----
with st.sidebar.expander("⚙️ Options", expanded=True):
    st.markdown("### Choose Language")
    language = st.radio("Languages", ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi"], key="language")

    st.markdown("### Choose Input Type")
    input_type = st.radio("Input Type", ["Enter Text", "Upload PDF", "Upload DOCX/TXT"], key="input_type")

if language != "Hindi":
    st.warning("Work in progress for this language.")
    st.stop()

# ----- Input Section -----
text = ""
if input_type == "Enter Text":
    text = st.text_area("Enter your text below:")
elif input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
elif input_type == "Upload DOCX/TXT":
    uploaded_file = st.file_uploader("Upload a DOCX or TXT file", type=["docx", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

# ----- Segment Button -----
if text:
    st.subheader("Your Given Text")
    st.write(text)

    if st.button("Segment Now"):
        import indicnlp
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        def split_sentences(text, lang='hi'):
            return sentence_split(text, lang)
    
        sentences = split_sentences(text)
        
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class SiameseContrastiveModel(nn.Module):
            def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
                super(SiameseContrastiveModel, self).__init__()
                self.encoder = AutoModel.from_pretrained(model_name)

            def forward(self, input_ids, attention_mask):
                output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                return output.last_hidden_state[:, 0, :]  # CLS token

        # Load model and tokenizer
        model = SiameseContrastiveModel().to(device)
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="Anirban1221/hindi_segmenter_generator", filename="segmenter_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        def get_sentence_embeddings(sentences, model, tokenizer):
            model.eval()
            model.to(device)
            embeddings = []

            with torch.no_grad():
                for i in range(0, len(sentences), 16):  # batch size
                    batch = sentences[i:i+16]
                    encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)

                    emb = model(input_ids, attention_mask)  # Uses forward()
                    embeddings.append(emb.cpu())

            return torch.cat(embeddings).numpy()
        
        embeddings = get_sentence_embeddings(sentences, model, tokenizer)


        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity

        def segment_text_graph(sentences, embeddings, sim_threshold=0.9):
            G = nx.Graph()

        # Add nodes for each sentence
            for i, sentence in enumerate(sentences):
                G.add_node(i, text=sentence)

        # Connect nodes (sentences) with edges based on similarity
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    if sim >= sim_threshold:
                        G.add_edge(i, j, weight=sim)

    # Detect communities (segments)
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))

    # Sort and return segments
            segments = []
            for community in sorted(communities, key=lambda x: min(x)):
                sorted_sents = [sentences[i] for i in sorted(community)]
                segments.append(" ".join(sorted_sents))

            return segments

# Test segmentation
        segments = segment_text_graph(sentences, embeddings)
        # for i, seg in enumerate(segments):
        #     print(f"[Segment {i+1}]: {seg}")

        
        st.subheader(" Segmented Output")

        for idx, seg in enumerate(segments, start=1):
            st.markdown(f"### Segment {idx}")
            st.markdown(f"{seg}\n")

