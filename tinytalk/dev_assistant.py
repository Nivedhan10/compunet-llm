"""Developer assistance and code base knowledge engine for TinyTalk.

Provides working, production-grade code examples, technical documentation references,
and developer assistance across Python, FastAPI, PyTorch, React, and databases.
"""
from __future__ import annotations

import re


DEV_TEMPLATES: dict[str, dict[str, str]] = {
    "python_crud": {
        "title": "Python CRUD (Create, Read, Update, Delete) Implementation",
        "doc_url": "https://docs.python.org/3/library/sqlite3.html",
        "doc_name": "Official Python Documentation (sqlite3 & data structures)",
        "content": (
            "Here is a complete, clean, and production-ready Python CRUD implementation using both an in-memory dictionary and a persistent SQLite database:\n\n"
            "```python\n"
            "# ====================================================================\n"
            "# 1. In-Memory Dictionary CRUD\n"
            "# ====================================================================\n"
            "class ItemStore:\n"
            "    def __init__(self):\n"
            "        self._db: dict[int, dict] = {}\n"
            "        self._next_id: int = 1\n\n"
            "    # CREATE\n"
            "    def create(self, name: str, price: float) -> dict:\n"
            "        item = {'id': self._next_id, 'name': name, 'price': price}\n"
            "        self._db[self._next_id] = item\n"
            "        self._next_id += 1\n"
            "        return item\n\n"
            "    # READ (single & all)\n"
            "    def get_by_id(self, item_id: int) -> dict | None:\n"
            "        return self._db.get(item_id)\n\n"
            "    def get_all(self) -> list[dict]:\n"
            "        return list(self._db.values())\n\n"
            "    # UPDATE\n"
            "    def update(self, item_id: int, name: str = None, price: float = None) -> dict | None:\n"
            "        item = self._db.get(item_id)\n"
            "        if not item:\n"
            "            return None\n"
            "        if name is not None:\n"
            "            item['name'] = name\n"
            "        if price is not None:\n"
            "            item['price'] = price\n"
            "        return item\n\n"
            "    # DELETE\n"
            "    def delete(self, item_id: int) -> bool:\n"
            "        return bool(self._db.pop(item_id, None))\n\n"
            "# Example Usage:\n"
            "store = ItemStore()\n"
            "created = store.create('Laptop', 999.99)  # CREATE\n"
            "print('Read:', store.get_by_id(1))       # READ\n"
            "store.update(1, price=899.99)             # UPDATE\n"
            "store.delete(1)                           # DELETE\n"
            "```\n\n"
            "```python\n"
            "# ====================================================================\n"
            "# 2. Persistent SQLite CRUD (Standard Library)\n"
            "# ====================================================================\n"
            "import sqlite3\n\n"
            "conn = sqlite3.connect(':memory:')  # or 'app.db'\n"
            "cursor = conn.cursor()\n\n"
            "# Create table\n"
            "cursor.execute('''\n"
            "    CREATE TABLE users (\n"
            "        id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "        username TEXT NOT NULL,\n"
            "        email TEXT UNIQUE NOT NULL\n"
            "    )\n"
            "''')\n"
            "conn.commit()\n\n"
            "# C - INSERT\n"
            "cursor.execute('INSERT INTO users (username, email) VALUES (?, ?)', ('Alice', 'alice@example.com'))\n"
            "conn.commit()\n\n"
            "# R - SELECT\n"
            "cursor.execute('SELECT * FROM users WHERE id = ?', (1,))\n"
            "user = cursor.fetchone()\n"
            "print('Fetched user:', user)\n\n"
            "# U - UPDATE\n"
            "cursor.execute('UPDATE users SET email = ? WHERE id = ?', ('alice_new@example.com', 1))\n"
            "conn.commit()\n\n"
            "# D - DELETE\n"
            "cursor.execute('DELETE FROM users WHERE id = ?', (1,))\n"
            "conn.commit()\n"
            "conn.close()\n"
            "```"
        )
    },
    "fastapi_crud": {
        "title": "FastAPI REST API CRUD Implementation",
        "doc_url": "https://fastapi.tiangolo.com/tutorial/body/",
        "doc_name": "Official FastAPI Documentation",
        "content": (
            "Here is a complete FastAPI REST API supporting all CRUD operations with Pydantic data validation:\n\n"
            "```python\n"
            "from fastapi import FastAPI, HTTPException, status\n"
            "from pydantic import BaseModel, Field\n\n"
            "app = FastAPI(title='CRUD API')\n\n"
            "class Item(BaseModel):\n"
            "    name: str = Field(min_length=1, max_length=100)\n"
            "    price: float = Field(gt=0)\n"
            "    in_stock: bool = True\n\n"
            "db: dict[int, dict] = {}\n"
            "counter = 1\n\n"
            "# CREATE\n"
            "@app.post('/items', status_code=status.HTTP_201_CREATED)\n"
            "def create_item(item: Item):\n"
            "    global counter\n"
            "    data = item.model_dump()\n"
            "    data['id'] = counter\n"
            "    db[counter] = data\n"
            "    counter += 1\n"
            "    return data\n\n"
            "# READ ALL & READ ONE\n"
            "@app.get('/items')\n"
            "def list_items():\n"
            "    return list(db.values())\n\n"
            "@app.get('/items/{item_id}')\n"
            "def get_item(item_id: int):\n"
            "    if item_id not in db:\n"
            "        raise HTTPException(status_code=404, detail='Item not found')\n"
            "    return db[item_id]\n\n"
            "# UPDATE\n"
            "@app.put('/items/{item_id}')\n"
            "def update_item(item_id: int, item: Item):\n"
            "    if item_id not in db:\n"
            "        raise HTTPException(status_code=404, detail='Item not found')\n"
            "    data = item.model_dump()\n"
            "    data['id'] = item_id\n"
            "    db[item_id] = data\n"
            "    return data\n\n"
            "# DELETE\n"
            "@app.delete('/items/{item_id}', status_code=status.HTTP_204_NO_CONTENT)\n"
            "def delete_item(item_id: int):\n"
            "    if item_id not in db:\n"
            "        raise HTTPException(status_code=404, detail='Item not found')\n"
            "    del db[item_id]\n"
            "```"
        )
    },
    "pytorch_training": {
        "title": "PyTorch Neural Network & Training Loop",
        "doc_url": "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
        "doc_name": "Official PyTorch Optimization Documentation",
        "content": (
            "Here is a complete PyTorch model definition and standard training loop from scratch:\n\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.optim as optim\n\n"
            "# 1. Define Model Architecture\n"
            "class SimpleMLP(nn.Module):\n"
            "    def __init__(self, in_features: int, hidden: int, out_features: int):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            "            nn.Linear(in_features, hidden),\n"
            "            nn.ReLU(),\n"
            "            nn.Dropout(0.1),\n"
            "            nn.Linear(hidden, out_features)\n"
            "        )\n\n"
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "        return self.net(x)\n\n"
            "# 2. Initialize Model, Loss, and Optimizer\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "model = SimpleMLP(in_features=10, hidden=32, out_features=2).to(device)\n"
            "criterion = nn.CrossEntropyLoss()\n"
            "optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)\n\n"
            "# 3. Standard Training Loop\n"
            "model.train()\n"
            "for epoch in range(5):\n"
            "    # Synthetic batch of 16 samples\n"
            "    inputs = torch.randn(16, 10, device=device)\n"
            "    targets = torch.randint(0, 2, (16,), device=device)\n\n"
            "    optimizer.zero_grad(set_to_none=True)  # Reset gradients\n"
            "    outputs = model(inputs)                # Forward pass\n"
            "    loss = criterion(outputs, targets)     # Loss computation\n"
            "    loss.backward()                        # Backpropagation\n"
            "    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
            "    optimizer.step()                       # Update parameters\n\n"
            "    print(f'Epoch {epoch+1} | Loss: {loss.item():.4f}')\n"
            "```"
        )
    },
    "python_file_io": {
        "title": "Python File Input/Output & JSON Handling",
        "doc_url": "https://docs.python.org/3/tutorial/inputoutput.html",
        "doc_name": "Official Python I/O Documentation",
        "content": (
            "Here is the standard, safe pattern for reading and writing files and JSON in Python:\n\n"
            "```python\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "file_path = Path('data.json')\n\n"
            "# Write structured data safely with UTF-8 encoding\n"
            "records = [\n"
            "    {'id': 1, 'task': 'Build model', 'done': True},\n"
            "    {'id': 2, 'task': 'Deploy API', 'done': False}\n"
            "]\n\n"
            "with open(file_path, 'w', encoding='utf-8') as f:\n"
            "    json.dump(records, f, indent=2, ensure_ascii=False)\n\n"
            "# Read data back\n"
            "if file_path.exists():\n"
            "    with open(file_path, 'r', encoding='utf-8') as f:\n"
            "        loaded = json.load(f)\n"
            "    print('Loaded:', loaded)\n"
            "```"
        )
    },
    "react_state_api": {
        "title": "React State Management & API Fetching",
        "doc_url": "https://react.dev/reference/react/useState",
        "doc_name": "Official React Documentation (Hooks)",
        "content": (
            "Here is a clean React functional component using `useState` and `useEffect` with asynchronous API fetching:\n\n"
            "```jsx\n"
            "import React, { useState, useEffect } from 'react';\n\n"
            "export default function UserList() {\n"
            "  const [users, setUsers] = useState([]);\n"
            "  const [loading, setLoading] = useState(true);\n"
            "  const [error, setError] = useState(null);\n\n"
            "  useEffect(() => {\n"
            "    const controller = new AbortController();\n"
            "    async function fetchData() {\n"
            "      try {\n"
            "        const response = await fetch('https://api.example.com/users', {\n"
            "          signal: controller.signal\n"
            "        });\n"
            "        if (!response.ok) throw new Error('Failed to fetch data');\n"
            "        const data = await response.json();\n"
            "        setUsers(data);\n"
            "      } catch (err) {\n"
            "        if (err.name !== 'AbortError') setError(err.message);\n"
            "      } finally {\n"
            "        setLoading(false);\n"
            "      }\n"
            "    }\n"
            "    fetchData();\n"
            "    return () => controller.abort(); // Cleanup on unmount\n"
            "  }, []);\n\n"
            "  if (loading) return <div>Loading...</div>;\n"
            "  if (error) return <div className='error'>{error}</div>;\n"
            "  return (\n"
            "    <ul>\n"
            "      {users.map(u => <li key={u.id}>{u.name}</li>)}\n"
            "    </ul>\n"
            "  );\n"
            "}\n"
            "```"
        )
    },
    "binary_search": {
        "title": "Binary Search Algorithm in Python",
        "doc_url": "https://docs.python.org/3/library/bisect.html",
        "doc_name": "Official Python Documentation (bisect module)",
        "content": (
            "Here is the iterative $O(\\log n)$ Binary Search implementation in Python:\n\n"
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    \"\"\"Return the index of target in sorted arr, or -1 if not found.\"\"\"\n"
            "    low = 0\n"
            "    high = len(arr) - 1\n\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n\n"
            "    return -1\n\n"
            "# Example:\n"
            "numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]\n"
            "idx = binary_search(numbers, 23)\n"
            "print('Found at index:', idx)  # 5\n"
            "```"
        )
    }
}


def find_developer_response(query: str) -> dict[str, str] | None:
    """Detect developer and programming queries, returning curated codebases and doc links."""
    q_lower = query.lower()

    # 1. Python CRUD operations
    if ("crud" in q_lower or ("create" in q_lower and "read" in q_lower and "delete" in q_lower)) and "fastapi" in q_lower:
        return DEV_TEMPLATES["fastapi_crud"]
    if "crud" in q_lower or ("create" in q_lower and "read" in q_lower and "update" in q_lower and "delete" in q_lower):
        return DEV_TEMPLATES["python_crud"]

    # 2. FastAPI
    if "fastapi" in q_lower and any(w in q_lower for w in ["code", "example", "endpoint", "api", "route", "crud"]):
        return DEV_TEMPLATES["fastapi_crud"]

    # 3. PyTorch training / model
    if "pytorch" in q_lower and any(w in q_lower for w in ["training", "train", "loop", "model", "loss", "optimizer", "code", "example"]):
        return DEV_TEMPLATES["pytorch_training"]

    # 4. File I/O & JSON
    if any(w in q_lower for w in ["file", "json", "read", "write"]) and any(w in q_lower for w in ["python", "code", "how to read", "how to write"]):
        if "file" in q_lower or "json" in q_lower:
            return DEV_TEMPLATES["python_file_io"]

    # 5. React hooks & State
    if "react" in q_lower and any(w in q_lower for w in ["state", "usestate", "useeffect", "hook", "fetch", "api", "code"]):
        return DEV_TEMPLATES["react_state_api"]

    # 6. Binary Search
    if "binary search" in q_lower:
        return DEV_TEMPLATES["binary_search"]

    return None

