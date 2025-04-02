
from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import PyPDF2
import io

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def extract_text_from_pdf_file(file_content):
    """Extract text from a PDF file content"""
    text = ""
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    docs = [Document(page_content=text)]
    chunks = text_splitter.split_documents(docs)
    return chunks

class RAGChatbot:


    def __init__(self, document_text=None):
        if document_text is not None:
            self.document = document_text
        else:
            with open('default_document.txt', 'r', encoding='utf-8') as f:
                self.document = f.read()
        
        self.chunks = split_text(self.document)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        texts = [chunk.page_content for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.texts = texts
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        if device == "cuda":
            model = model.to(device)
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15
        )
            
    def retrieve_context(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        _, indices = self.faiss_index.search(query_embedding, top_k)
        
        retrieved_chunks = [self.texts[idx] for idx in indices[0]]
        
        context = "\n\n".join(retrieved_chunks)
        return context
    
    def answer(self, query):
        context = self.retrieve_context(query)
        prompt = f"""<|system|>
You are a helpful assistant specialized in the knowledge from the provided documents. Use the following context to answer the question, and if you don't know the answer from the context, just say you don't know.

Context:
{context}

<|user|>
{query}

<|assistant|>
"""
        
        response = self.llm_pipeline(prompt)[0]['generated_text']
        
        assistant_response = response.split("<|assistant|>")[-1].strip()
        return assistant_response
    
    def update_knowledge(self, document_text):
        """Update the chatbot's knowledge with new document"""
        self.document = document_text
        
        self.chunks = split_text(self.document)
        
        texts = [chunk.page_content for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.texts = texts
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
        
        return True


chatbot = RAGChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    
    if not query:
        return jsonify({'response': 'Please ask a question.'})
    
    response = chatbot.answer(query)
    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'status': 'error', 'message': 'Only PDF files are supported'})
    
    try:
        file_content = file.read()
        
        document_text = extract_text_from_pdf_file(file_content)
        
        success = chatbot.update_knowledge(document_text)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Successfully uploaded and processed {file.filename}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update chatbot knowledge'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}'
        })

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)
