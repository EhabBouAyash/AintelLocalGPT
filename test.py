from sentence_transformers import SentenceTransformer
from constants import EMBEDDING_MODEL_NAME, SOURCE_DIRECTORY
from langchain.embeddings import HuggingFaceEmbeddings
from ctransformers import AutoModelForCausalLM
from langchain import PromptTemplate
from ingest import split_documents, load_documents
import logging
from transformers import AutoTokenizer, AutoModel
# Load pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Example input sentence
logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
documents = load_documents(SOURCE_DIRECTORY)
text_documents, python_documents = split_documents(documents)

# Generate embedding for the sentence
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF")
print(model("Who is Nabih Berry?"))
