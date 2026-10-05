# Multimodal RAG Question-Answering System · Demo Project

A runnable, pure-frontend demo that requires no backend and showcases the full "Multimodal RAG + Agent" interaction pipeline.

## Quick Start

```bash
npm install      # or pnpm install
npm run dev      # open http://localhost:3000
```

> Requires Node.js 18.18+ (developed on Node 22).

## Demo Mode

- **Demo mode is on by default** (`DEMO_MODE=true`): no real backend, LLM, or database is connected.
- The built-in mock engine (`lib/store.ts`) simulates the full agent pipeline and streams a "reasoning" timeline step by step:

  Intent analysis → Knowledge-base retrieval → Relevance judgment → Web crawling → OCR extraction → Answer generation

- Answers branch by the question's keywords to describe different modules, and return retrieval/generation latency to drive the "Analytics dashboard" stats and trend chart.

### Switching to a real backend

Edit `.env.local`:

```bash
# turn off demo mode
DEMO_MODE=false
# uncomment and point to the real RAG backend
BACKEND_URL=http://localhost:8001
```

## Feature Demo Guide

| Capability | Example prompt |
|---|---|
| Vector retrieval / RAG | `What is vector retrieval?` |
| Multimodal | `Explain multimodal technology` |
| Agent | `How does the agent work?` |
| Web crawler | `How does the crawler fetch the latest papers?` |
| OCR | `How does OCR extract text from documents?` |
| Self-improvement | `What is the self-improvement mechanism?` |

Each answer shows the "Agent reasoning process" timeline above it; use the top-right "Dashboard" for message stats and latency trends, and "Settings" to adjust reasoning-chain length and temperature.

## Directory Structure

```
frontend/
├── app/
│   ├── api/            # API routes (in demo mode /api/v1/* uses the mock engine)
│   └── page.tsx        # home page
├── components/
│   └── multimodal-chat.tsx   # main chat component
├── lib/
│   └── store.ts        # mock task engine (simulates the full agent pipeline)
└── .env.local          # demo-mode switch
```

## Notes

- The real backend lives in other directories of the repo: `rag/` (FastAPI RAG backend), `backend/` (demo backend), `crawler/` (arXiv crawler), `ocr/` (OCR), `query-relevance/` (query rewriting & relevance judgment).
- Running the real backend locally needs an NVIDIA GPU (vLLM) plus an 8B model; demo mode needs no extra resources.
