# RAG Pipeline — Bug Review

> **Code review for a production RAG pipeline submitted by a junior engineer.**
> 6 bugs identified across 3 severity categories: runtime crashes, silent wrong answers, and architectural flaws.

---

## Summary

| Category | Count |
|---|---|
| Runtime crashes | 2 |
| Silent wrong answers | 2 |
| Architectural flaws | 2 |
| **Total bugs** | **6** |

---

## Bug 01 — Wrong embedding model name

**Category:** Runtime crash

**What it causes:**
The code passes `model="gpt-4"` to `OpenAIEmbeddings`. GPT-4 is a chat completion model — OpenAI's embedding API rejects it immediately with a 400 error. The pipeline crashes on startup before any document is processed.

**Fix:**
```python
# WRONG
embeddings = OpenAIEmbeddings(model="gpt-4")

# CORRECT
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

---

## Bug 02 — No error handling on query

**Category:** Runtime crash

**What it causes:**
Any transient failure — rate limit, network timeout, API outage — raises an unhandled exception and terminates the process. In production this silently kills the service with no log, no retry, and no user-facing message.

**Fix:**
```python
# WRONG
result = qa_chain.run("What is our refund policy?")

# CORRECT
try:
    result = qa_chain.invoke({"query": "What is our refund policy?"})
    print("Answer:", result["result"])
except Exception as e:
    print(f"Query failed: {e}")
```

---

## Bug 03 — chunk_overlap larger than chunk_size

**Category:** Silent wrong answers

**What it causes:**
With `chunk_size=100` and `chunk_overlap=500`, overlap exceeds the chunk itself. The splitter produces heavily duplicated, garbled fragments. No error is raised — the pipeline runs and returns answers, but they are based on mangled repeated text rather than coherent passages.

**Fix:**
```python
# WRONG
CharacterTextSplitter(chunk_size=100, chunk_overlap=500)

# CORRECT
CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

---

## Bug 04 — temperature=1.0 on a factual QA chain

**Category:** Silent wrong answers

**What it causes:**
Temperature 1.0 maximises randomness. For a factual retrieval system this means the model paraphrases, embellishes, or invents details that are not in the source documents — and does so differently on every call. The answers look plausible but are unreliable. No exception is raised.

**Fix:**
```python
# WRONG
ChatOpenAI(model="gpt-4", temperature=1.0)

# CORRECT
ChatOpenAI(model="gpt-4", temperature=0)
```

---

## Bug 05 — k=50 retrieved chunks causes context overflow

**Category:** Architectural flaw

**What it causes:**
Retrieving 50 chunks at `chunk_size=100` means up to 5,000 tokens of context per query — but with a realistic chunk size of 1,000 tokens that becomes 50,000 tokens, far beyond GPT-4's context window. Under load this causes truncation errors or API rejections. The bug is invisible in small tests but fails at scale.

**Fix:**
```python
# WRONG
search_kwargs={"k": 50}

# CORRECT
search_kwargs={"k": 4}
```

---

## Bug 06 — return_source_documents=False

**Category:** Architectural flaw

**What it causes:**
Disabling source document return means there is no way to audit which passages the answer came from. In production this makes hallucination detection impossible, breaks any citation UI, and removes the ability to debug why the model gave a wrong answer. It works fine in demos but is unshippable in a real system.

**Fix:**
```python
# WRONG
return_source_documents=False

# CORRECT
return_source_documents=True
```

---

## Corrected Code — Full Pipeline

```python
import os
os.environ["GROQ_API_KEY"] = "your-key-here"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. Load documents
loader = TextLoader("company_docs.txt")
docs = loader.load()

# 2. Split into chunks — size must be larger than overlap
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Embeddings — use an actual embedding model, not a chat model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # k=4 not 50

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know, say "I don't have that information."

Context: {context}

Question: {question}
""")

# 5. LLM — temperature=0 for factual answers
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query with error handling
try:
    result = chain.invoke("What is our refund policy?")
    print("Answer:", result)
except Exception as e:
    print(f"Query failed: {e}")
```

---

## Quick Reference — All Fixes

| # | Bug | Type | Effect if not fixed |
|---|-----|------|---------------------|
| 01 | `model="gpt-4"` in embeddings | Runtime crash | API error on startup |
| 02 | No try/except on query | Runtime crash | Silent process termination |
| 03 | `overlap > chunk_size` | Silent wrong answers | Garbled, repeated chunks |
| 04 | `temperature=1.0` | Silent wrong answers | Hallucinated, non-deterministic answers |
| 05 | `k=50` chunks retrieved | Architectural | Context window overflow under load |
| 06 | `return_source_documents=False` | Architectural | No auditability or hallucination detection |
