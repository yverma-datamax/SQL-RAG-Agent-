"""
rag_utils.py — Core RAG components:
  - SchemaRetriever  : embeds schema chunks and retrieves relevant tables
  - SQLGenerator     : calls Ollama to turn a question + schema into SQL
  - QueryExecutor    : runs SQL against SQLite safely
  - SQLRAGAgent      : full pipeline with self-correction loop
"""

import sqlite3
import re
import numpy as np
import requests
from typing import Optional


# ── 1. SCHEMA RETRIEVER ─────────────────────────────────────────────────────

class SchemaRetriever:
    """
    Converts each table's schema into a text chunk, embeds them with
    nomic-embed-text (via Ollama), and retrieves the top-k most relevant
    tables for a given natural-language question.

    This is the 'R' in RAG — pure vector similarity, no magic.
    """

    def __init__(self, db_path: str, ollama_url: str = "http://localhost:11434"):
        self.db_path = db_path
        self.ollama_url = ollama_url
        self.chunks: list[str] = []
        self.embeddings: np.ndarray = None
        self._build_schema_chunks()

    def _build_schema_chunks(self):
        """Read the live database schema and turn each table into a text chunk."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]

        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            cols = cur.fetchall()

            cur.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = cur.fetchall()

            col_desc = ", ".join(
                f"{col[1]} {col[2]}{'(PK)' if col[5] else ''}"
                for col in cols
            )
            sample_text = "\n".join(str(row) for row in sample_rows)

            chunk = (
                f"Table: {table}\n"
                f"Columns: {col_desc}\n"
                f"Sample rows:\n{sample_text}"
            )
            self.chunks.append(chunk)

        conn.close()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Call Ollama's embedding endpoint for a batch of texts."""
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return np.array(vectors, dtype=np.float32)

    def build_index(self):
        """Embed all schema chunks. Call once before querying."""
        print(f"Embedding {len(self.chunks)} table schemas...")
        self.embeddings = self.embed(self.chunks)
        print("Index ready.")

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        """
        Embed the question, compute cosine similarity against all table
        embeddings, and return the top_k most relevant schema chunks.
        top_k=3 returns ALL tables, ensuring JOINs are always possible.
        """
        if self.embeddings is None:
            raise RuntimeError("Call build_index() first.")

        q_vec = self.embed([question])[0]

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        scores = [cosine_sim(q_vec, e) for e in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]


# ── 2. SQL GENERATOR ─────────────────────────────────────────────────────────

class SQLGenerator:
    """
    Sends the retrieved schema context + user question to a local LLM
    (via Ollama) and extracts a SQL query from the response.

    This is the 'A + G' in RAG — augment the prompt, then generate.
    """

    SYSTEM_PROMPT = """You are an expert SQLite SQL assistant.

Rules:
- Return ONLY the SQL query, nothing else — no explanation, no markdown fences
- Use ONLY SQLite-compatible syntax
- Use strftime('%Y-%m', date_column) for date grouping — never use EXTRACT()
- Always JOIN tables when columns from multiple tables are needed
- Use table and column names exactly as shown in the schema
- Alias aggregated columns clearly (e.g. SUM(total_amount) AS revenue)"""

    CORRECTION_PROMPT = """The SQL query you wrote caused this error:

Error: {error}

Your query was:
{sql}

Fix the query and return ONLY the corrected SQL. No explanation."""

    def __init__(self, model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url

    def _call_ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "system": self.SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        return re.sub(r"```sql|```", "", raw).strip()

    def generate(self, question: str, schema_chunks: list[str]) -> str:
        """Generate SQL from a question and schema context."""
        schema_block = "\n\n".join(schema_chunks)
        prompt = f"Schema:\n{schema_block}\n\nQuestion: {question}\n\nSQL:"
        return self._call_ollama(prompt)

    def correct(self, sql: str, error: str) -> str:
        """
        Self-correction — feed the failed SQL + error message back to the LLM
        and ask it to fix the query. This is the key improvement over v1.
        """
        prompt = self.CORRECTION_PROMPT.format(sql=sql, error=error)
        return self._call_ollama(prompt)


# ── 3. QUERY EXECUTOR ────────────────────────────────────────────────────────

class QueryExecutor:
    """
    Safely executes a SELECT query against SQLite.
    Rejects any non-SELECT statement.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, sql: str) -> tuple[list[dict], Optional[str]]:
        if not sql.strip().upper().startswith("SELECT"):
            return [], "Only SELECT queries are allowed."
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql)
            rows = [dict(row) for row in cur.fetchall()]
            conn.close()
            return rows, None
        except sqlite3.Error as e:
            return [], str(e)


# ── 4. FULL PIPELINE WITH SELF-CORRECTION ────────────────────────────────────

class SQLRAGAgent:
    """
    End-to-end RAG pipeline with self-correction:
      question
        → retrieve relevant schema (top_k=3, all tables)
        → generate SQL
        → execute
        → if error: correct SQL using the error message and retry
        → return results
    """

    def __init__(self, db_path: str, model: str = "mistral",
                 ollama_url: str = "http://localhost:11434"):
        self.retriever = SchemaRetriever(db_path, ollama_url)
        self.generator = SQLGenerator(model, ollama_url)
        self.executor = QueryExecutor(db_path)

    def build_index(self):
        self.retriever.build_index()

    def ask(self, question: str, top_k: int = 3, max_retries: int = 2,
            verbose: bool = True) -> dict:
        """
        Run the full RAG pipeline.
        If SQL execution fails, automatically retry with self-correction.
        """

        # Step 1 — Retrieve
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # Step 2 — Generate
        sql = self.generator.generate(question, chunks)

        # Step 3 — Execute with self-correction loop
        attempts = []
        rows, error = self.executor.execute(sql)
        attempts.append({"sql": sql, "error": error})

        retry = 0
        while error and retry < max_retries:
            if verbose:
                print(f"  Attempt {retry + 1} failed: {error}")
                print(f"  Retrying with self-correction...")
            sql = self.generator.correct(sql, error)
            rows, error = self.executor.execute(sql)
            attempts.append({"sql": sql, "error": error})
            retry += 1

        if verbose:
            print(f"Question : {question}")
            print(f"Retrieved: {len(chunks)} table(s)")
            print(f"SQL      :\n{sql}")
            if error:
                print(f"Failed after {retry} correction(s): {error}")
            else:
                n_attempts = len(attempts)
                label = "first try" if n_attempts == 1 else f"after {n_attempts - 1} correction(s)"
                print(f"Success ({label}) — {len(rows)} rows returned")

        return {
            "question": question,
            "retrieved_schema": chunks,
            "sql": sql,
            "rows": rows,
            "error": error,
            "attempts": attempts,
        }