# ─────────────────────────────────────────────────────────────────────────────
# ClinIQ — Makefile
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: setup run-api run-frontend run-all docker-up docker-down test clean help

help:
	@echo ""
	@echo "  ClinIQ — Available Commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make setup        Install dependencies & setup env"
	@echo "  make run-api      Start FastAPI backend (localhost:8000)"
	@echo "  make run-frontend Start Streamlit frontend (localhost:8501)"
	@echo "  make run-all      Start both API and frontend"
	@echo "  make docker-up    Build & run with Docker Compose"
	@echo "  make docker-down  Stop Docker containers"
	@echo "  make test         Run tests"
	@echo "  make clean        Remove generated data"
	@echo ""

setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "📋 Setting up .env..."
	@if [ ! -f .env ]; then cp .env.example .env && echo "✅ .env created — add your GROQ_API_KEY"; fi
	@mkdir -p data/chroma_db data/checkpoints data/samples
	@echo "✅ Setup complete!"

run-api:
	@echo "🚀 Starting ClinIQ API on http://localhost:8000"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

run-frontend:
	@echo "🎨 Starting ClinIQ Frontend on http://localhost:8501"
	streamlit run frontend/app.py --server.port 8501

run-all:
	@echo "🚀 Starting ClinIQ (API + Frontend)..."
	@make run-api &
	@sleep 5
	@make run-frontend

docker-up:
	@echo "🐳 Building and starting with Docker..."
	docker compose up --build

docker-down:
	@echo "🛑 Stopping containers..."
	docker compose down

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --tb=short

clean:
	@echo "🧹 Cleaning generated data..."
	rm -rf data/chroma_db data/checkpoints
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete"
