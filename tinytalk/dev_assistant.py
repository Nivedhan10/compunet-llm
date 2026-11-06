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
    },
    "django_cli": {
        "title": "Django CLI, Project Setup & Server Commands",
        "doc_url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/",
        "doc_name": "Official Django Documentation",
        "content": (
            "Here are the standard terminal commands to create, configure, migrate, and start a Django application:\n\n"
            "```bash\n"
            "# 1. Install Django in your virtual environment\n"
            "pip install django\n\n"
            "# 2. Create a new Django project\n"
            "django-admin startproject myproject\n"
            "cd myproject\n\n"
            "# 3. Create a Django application within the project\n"
            "python manage.py startapp myapp\n\n"
            "# 4. Generate and apply database migrations\n"
            "python manage.py makemigrations\n"
            "python manage.py migrate\n\n"
            "# 5. Start the development server (default: http://127.0.0.1:8000/)\n"
            "python manage.py runserver\n\n"
            "# Optional: Run on a custom port or bind to all interfaces\n"
            "# python manage.py runserver 0.0.0.0:8000\n"
            "```\n\n"
            "```python\n"
            "# Minimal views.py in myapp:\n"
            "from django.http import HttpResponse\n\n"
            "def index(request):\n"
            "    return HttpResponse('Hello from Django!')\n"
            "```"
        )
    },
    "flask_cli": {
        "title": "Flask Application & Server Setup",
        "doc_url": "https://flask.palletsprojects.com/en/stable/quickstart/",
        "doc_name": "Official Flask Documentation",
        "content": (
            "Here is a complete Flask API application and the terminal commands to run it:\n\n"
            "```python\n"
            "# app.py\n"
            "from flask import Flask, jsonify, request\n\n"
            "app = Flask(__name__)\n\n"
            "@app.route('/')\n"
            "def home():\n"
            "    return jsonify({'message': 'Flask server is running!', 'status': 200})\n\n"
            "if __name__ == '__main__':\n"
            "    app.run(debug=True, port=5000)\n"
            "```\n\n"
            "```bash\n"
            "# Terminal commands:\n"
            "pip install flask\n"
            "flask --app app run --debug\n"
            "# or directly:\n"
            "python app.py\n"
            "```"
        )
    },
    "docker_cli": {
        "title": "Docker Container Build & Run Commands",
        "doc_url": "https://docs.docker.com/get-started/",
        "doc_name": "Official Docker Documentation",
        "content": (
            "Here are the essential Docker commands to build and run containerized services:\n\n"
            "```dockerfile\n"
            "# Dockerfile\n"
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8000\n"
            "CMD [\"uvicorn\", \"tinytalk.api:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            "```\n\n"
            "```bash\n"
            "# 1. Build image\n"
            "docker build -t myapp:latest .\n\n"
            "# 2. Run container mapping port 8000\n"
            "docker run -d -p 8000:8000 --name myapp_container myapp:latest\n\n"
            "# 3. Check logs and running containers\n"
            "docker ps\n"
            "docker logs -f myapp_container\n\n"
            "# 4. Stop and remove container\n"
            "docker stop myapp_container\n"
            "docker rm myapp_container\n"
            "```"
        )
    },
    "git_cli": {
        "title": "Git Version Control Workflow & CLI Commands",
        "doc_url": "https://git-scm.com/doc",
        "doc_name": "Official Git Documentation",
        "content": (
            "Here are the core Git terminal commands for everyday development:\n\n"
            "```bash\n"
            "# 1. Clone repository or initialize new\n"
            "git clone <repository_url>\n"
            "git init\n\n"
            "# 2. Inspect workspace\n"
            "git status\n"
            "git diff\n\n"
            "# 3. Stage changes and commit\n"
            "git add .\n"
            "git commit -m \"feat: implement new capability\"\n\n"
            "# 4. Branching\n"
            "git branch -M main\n"
            "git checkout -b feature/feature-name\n\n"
            "# 5. Push changes to remote\n"
            "git push -u origin feature/feature-name\n"
            "```"
        )
    },
    "node_vite_cli": {
        "title": "Node.js & Vite Frontend Development Commands",
        "doc_url": "https://vite.dev/guide/",
        "doc_name": "Official Vite Documentation",
        "content": (
            "Here are the standard terminal commands to create and run a React Vite frontend:\n\n"
            "```bash\n"
            "# 1. Scaffold a React project with Vite\n"
            "npm create vite@latest my-app -- --template react\n"
            "cd my-app\n\n"
            "# 2. Install dependencies\n"
            "npm install\n\n"
            "# 3. Start development server with Hot Module Replacement (HMR)\n"
            "npm run dev\n\n"
            "# 4. Build optimized production bundle\n"
            "npm run build\n"
            "npm run preview\n"
            "```"
        )
    },
    "java_basics": {
        "title": "Java Object-Oriented Programming & CLI Execution",
        "doc_url": "https://docs.oracle.com/en/java/",
        "doc_name": "Official Oracle Java Documentation",
        "content": (
            "Here is a complete, clean Java class demonstration with object-oriented encapsulation and compilation instructions:\n\n"
            "```java\n"
            "// Main.java\n"
            "public class Main {\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(\"Hello from Java!\");\n\n"
            "        // Object-Oriented Instance\n"
            "        Developer dev = new Developer(\"Alex\", \"Java\");\n"
            "        dev.displayInfo();\n"
            "    }\n"
            "}\n\n"
            "class Developer {\n"
            "    private String name;\n"
            "    private String language;\n\n"
            "    public Developer(String name, String language) {\n"
            "        this.name = name;\n"
            "        this.language = language;\n"
            "    }\n\n"
            "    public void displayInfo() {\n"
            "        System.out.printf(\"Developer: %s | Primary Tech: %s%n\", name, language);\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "```bash\n"
            "# Terminal compilation and execution commands:\n"
            "# 1. Compile source into bytecode (.class file)\n"
            "javac Main.java\n\n"
            "# 2. Run the bytecode on the Java Virtual Machine (JVM)\n"
            "java Main\n"
            "```"
        )
    }
}


def find_developer_response(query: str) -> dict[str, str] | None:
    """Detect developer and programming queries, returning curated codebases and doc links."""
    q_lower = query.lower()

    # Java programming code / syntax
    if "java" in q_lower.split() and any(w in q_lower for w in ["code", "example", "syntax", "class", "main", "compile", "javac", "oop", "program"]):
        return DEV_TEMPLATES["java_basics"]

    # 1. Django commands & setup
    if "django" in q_lower and any(w in q_lower for w in [
        "start", "run", "command", "server", "cli", "project", "app", "manage.py",
        "terminal", "setup", "create", "how to start", "how to run"
    ]):
        return DEV_TEMPLATES["django_cli"]

    # 2. Flask setup & API
    if "flask" in q_lower and any(w in q_lower for w in [
        "start", "run", "command", "server", "code", "app", "terminal", "setup", "api"
    ]):
        return DEV_TEMPLATES["flask_cli"]

    # 3. Docker commands
    if "docker" in q_lower and any(w in q_lower for w in [
        "build", "run", "command", "container", "image", "terminal", "dockerfile", "start", "stop", "compose"
    ]):
        return DEV_TEMPLATES["docker_cli"]

    # 4. Git workflow & commands
    if any(k in q_lower for k in ["git commit", "git push", "git pull", "git status", "git clone", "git branch", "git command", "git workflow"]) or (
        "git" in q_lower.split() and any(w in q_lower for w in ["command", "terminal", "commit", "push", "pull", "clone", "branch", "repo"])
    ):
        return DEV_TEMPLATES["git_cli"]

    # 5. Node.js / Vite / NPM
    if any(k in q_lower for k in ["vite", "npm create", "npm run dev", "npm install", "vite dev"]) or (
        "npm" in q_lower and any(w in q_lower for w in ["command", "start", "run", "dev", "build"])
    ):
        return DEV_TEMPLATES["node_vite_cli"]

    # 6. Python CRUD operations
    if ("crud" in q_lower or ("create" in q_lower and "read" in q_lower and "delete" in q_lower)) and "fastapi" in q_lower:
        return DEV_TEMPLATES["fastapi_crud"]
    if "crud" in q_lower or ("create" in q_lower and "read" in q_lower and "update" in q_lower and "delete" in q_lower):
        return DEV_TEMPLATES["python_crud"]

    # 7. FastAPI
    if "fastapi" in q_lower and any(w in q_lower for w in ["code", "example", "endpoint", "api", "route", "crud"]):
        return DEV_TEMPLATES["fastapi_crud"]

    # 8. PyTorch training / model
    if "pytorch" in q_lower and any(w in q_lower for w in ["training", "train", "loop", "model", "loss", "optimizer", "code", "example"]):
        return DEV_TEMPLATES["pytorch_training"]

    # 9. File I/O & JSON
    if any(w in q_lower for w in ["file", "json", "read", "write"]) and any(w in q_lower for w in ["python", "code", "how to read", "how to write"]):
        if "file" in q_lower or "json" in q_lower:
            return DEV_TEMPLATES["python_file_io"]

    # 10. React hooks & State
    if "react" in q_lower and any(w in q_lower for w in ["state", "usestate", "useeffect", "hook", "fetch", "api", "code"]):
        return DEV_TEMPLATES["react_state_api"]

    # 11. Binary Search
    if "binary search" in q_lower:
        return DEV_TEMPLATES["binary_search"]

    return None

