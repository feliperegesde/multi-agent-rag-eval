import os
import sqlite3
import urllib.request
import json
from pathlib import Path

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
DB_PATH = DATA_DIR / "fastapi_metadata.db"

# Páginas centrais da doc oficial em Markdown
DOC_FILES = [
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/dependencies/index.md",
    "tutorial/security/index.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "tutorial/background-tasks.md",
    "deployment/docker.md"
]

BASE_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/"

def setup_docs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print("📥 Baixando documentação técnica do FastAPI...")
    for rel_path in DOC_FILES:
        target_path = DOCS_DIR / Path(rel_path).name
        url = BASE_URL + rel_path
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  ✓ Salvo: {target_path.name}")
        except Exception as e:
            print(f"  ✗ Erro ao baixar {rel_path}: {e}")

def setup_sqlite():
    print("\n📦 Criando banco SQLite com histórico de releases...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS releases (
            version TEXT PRIMARY KEY,
            release_date TEXT,
            python_min_version TEXT,
            key_features TEXT
        )
    """)
    
    # Releases históricas e reais para o Text-to-SQL
    releases_data = [
        ("0.115.0", "2024-09-18", "3.8", "Support for Pydantic v2.9, upgraded Starlette to 0.38.5"),
        ("0.114.0", "2024-08-30", "3.8", "Refactored OpenAPI schema generation, improved lifespan events"),
        ("0.112.0", "2024-08-10", "3.8", "Added support for background task callbacks and validation error tuning"),
        ("0.111.0", "2024-05-15", "3.8", "Improved typing annotations, FastAPI CLI integrations"),
        ("0.109.0", "2024-01-12", "3.8", "Drop Python 3.7 support, require Python 3.8+"),
        ("0.100.0", "2023-07-08", "3.7", "Full official support for Pydantic v2 with backwards compatibility"),
        ("0.95.0", "2023-03-24", "3.7", "New dependency injection performance improvements")
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO releases (version, release_date, python_min_version, key_features)
        VALUES (?, ?, ?, ?)
    """, releases_data)
    
    conn.commit()
    conn.close()
    print(f"  ✓ Banco criado em: {DB_PATH}")

if __name__ == "__main__":
    setup_docs()
    setup_sqlite()