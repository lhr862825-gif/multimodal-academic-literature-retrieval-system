export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Task {
  id: string;
  status: TaskStatus;
  query: string;
  result?: any;
  error?: string;
  createdAt: number;
}

// In demo mode this simulates the real RAG + Agent pipeline and branches the
// answer based on the question's keywords.
function buildMockAnswer(query: string): { answer: string; sources: string[]; confidence: number } {
  const q = query.toLowerCase();

  if (q.includes('multimodal')) {
    return {
      answer:
        `Multimodal AI is a technology that can simultaneously process and understand multiple data types (text, images, audio, video, etc.).\n\n**Core principles:**\n1. Cross-modal understanding: recognizing semantic relationships across different data types\n2. Unified representation: mapping different modalities into a shared feature space\n3. Synergistic enhancement: modalities complement each other to improve overall comprehension\n\n**In this project**, multimodal capability is realized through support for text, image, and file inputs, where an OCR module converts information in images/PDFs into searchable text that is then indexed into a vector knowledge base for semantic Q&A.`,
      sources: ['Knowledge base doc A (multimodal survey)', 'arXiv:2401.xxxxx.pdf'],
      confidence: 0.95,
    };
  }

  if (q.includes('vector') || q.includes('retrieval') || q.includes('rag') || q.includes('embedding')) {
    return {
      answer:
        `Vector retrieval (the core of RAG) is an information retrieval technique based on semantic similarity.\n\n**How it works:**\n1. Text embedding: an embedding model (e.g. bge-m3) converts text into high-dimensional vectors\n2. Similarity scoring: the query and documents are compared in the vector space\n3. Ranking: the most relevant document chunks are returned by similarity score\n\nUnlike traditional keyword search, vector retrieval understands semantics and finds content that is "phrased differently but means the same thing". This project uses **hybrid retrieval** (vector search + BM25 keyword search) to balance semantic and exact matching.`,
      sources: ['Knowledge base doc B (vector retrieval)', 'Local knowledge base'],
      confidence: 0.93,
    };
  }

  if (q.includes('learning') || q.includes('self') || q.includes('grow') || q.includes('feedback')) {
    return {
      answer:
        `Self-improvement refers to an AI system's ability to continuously learn and optimize from user interactions and feedback.\n\n**Core mechanisms:**\n1. Feedback collection: recording user satisfaction and corrections\n2. Model fine-tuning: adjusting model parameters with new data\n3. Knowledge updates: automatically expanding and refreshing the knowledge base\n4. Quality monitoring: evaluating answer quality in real time and optimizing accordingly\n\nThis lets the system "get smarter with use" and adapt to specific business scenarios.`,
      sources: ['Knowledge base doc C (self-improvement)', 'User feedback log'],
      confidence: 0.9,
    };
  }

  if (q.includes('crawler') || q.includes('crawl') || q.includes('arxiv') || q.includes('paper')) {
    return {
      answer:
        `The web crawler module is the system's "eyes", triggered automatically when the local knowledge base cannot answer a question.\n\n**Execution flow:**\n1. Relevance judgment: the agent evaluates whether local retrieval results are sufficient\n2. Web search: the latest papers are fetched from academic sources such as arXiv\n3. Download & parse: PDFs are downloaded and passed to the OCR module for text extraction\n4. Index update: new content is written into the vector knowledge base for future retrieval\n\nThis allows the system to answer questions beyond its knowledge base, forming a self-replenishing loop.`,
      sources: ['arXiv:2401.xxxxx.pdf (newly crawled)'],
      confidence: 0.88,
    };
  }

  if (q.includes('ocr') || q.includes('recognize') || q.includes('extract')) {
    return {
      answer:
        `OCR (Optical Character Recognition) extracts text from images, scanned documents, and PDFs.\n\n**OCR capabilities in this project:**\n1. Powered by PaddleOCR (PP-OCRv5 detection + recognition models)\n2. Supports Chinese and English recognition\n3. Extracted text is converted to Markdown, then chunked, embedded, and added to the knowledge base\n\nThis makes the content inside images and documents searchable and answerable — a key step in enabling multimodal input.`,
      sources: ['Local document (OCR-parsed result)'],
      confidence: 0.91,
    };
  }

  if (q.includes('agent')) {
    return {
      answer:
        `An agent is an AI system with autonomous decision-making and tool-use capabilities.\n\n**This project's agent pipeline (Corrective RAG):**\n1. Intent analysis: understand the user's question\n2. Knowledge-base retrieval: first query the local vector store\n3. Relevance judgment: assess whether the retrieved results are sufficient\n4. Autonomous decision: if insufficient, trigger the web crawler + OCR\n5. Answer generation: synthesize all information into a final answer\n\nEvery step streams a timeline back to the UI, so you can watch the "agent reasoning process" in real time.`,
      sources: ['Agent execution log', 'Knowledge base doc A'],
      confidence: 0.94,
    };
  }

  return {
    answer:
      `I understand your question is about "${query}". Based on local knowledge-base retrieval and the agent's autonomous judgment, here is a summary:\n\nThis is a **multimodal RAG question-answering system** with the following capabilities:\n\n1. **Multimodal input**: supports text, images, and files (images/PDFs go through OCR)\n2. **Vector retrieval**: embedding + hybrid search (vector + BM25) to locate relevant knowledge\n3. **Agent decision-making**: automatically triggers the web crawler to fetch the latest papers when local knowledge is insufficient\n4. **Self-improvement**: continuously optimizes answer quality from user feedback\n\nYou can watch the full pipeline in the "Agent reasoning process" panel. To explore a specific module, try asking about "vector retrieval", "multimodal", "OCR", "crawler", or "agent".`,
    sources: ['Local knowledge base', 'System documentation'],
    confidence: 0.85,
  };
}

// Global in-memory store
class TaskStore {
  private tasks: Map<string, Task> = new Map();

  createTask(query: string): Task {
    const id = Date.now().toString() + Math.random().toString(36).substring(7);
    const task: Task = {
      id,
      status: 'pending',
      query,
      createdAt: Date.now(),
    };
    this.tasks.set(id, task);

    // Simulate async processing
    this.processTask(id);

    return task;
  }

  getTask(id: string): Task | undefined {
    return this.tasks.get(id);
  }

  private async processTask(id: string) {
    const task = this.tasks.get(id);
    if (!task) return;

    // Update to processing
    task.status = 'processing';
    // @ts-ignore
    task.timeline = [];
    this.tasks.set(id, task);

    // Helper to add a timeline event
    const addEvent = (event: string, message: string, detail?: string) => {
      // @ts-ignore
      if (!task.timeline) task.timeline = [];
      // @ts-ignore
      task.timeline.push({
        timestamp: Date.now(),
        event,
        message,
        detail,
      });
      this.tasks.set(id, task);
    };

    // A simple keyword hint shown on the timeline
    const hint = task.query.trim().slice(0, 24) || 'n/a';

    // 1. Query Analysis
    await new Promise((resolve) => setTimeout(resolve, 900));
    addEvent('query_analysis', 'Analyzing the question intent...', `Extracted keywords: ${hint}`);

    // 2. Retrieval
    await new Promise((resolve) => setTimeout(resolve, 1200));
    addEvent('retrieval', 'Searching the knowledge base...', 'Hybrid retrieval (vector + BM25) matched 3 relevant documents');

    // 3. Judgment
    await new Promise((resolve) => setTimeout(resolve, 1200));
    addEvent('judgment', 'Agent judgment: local resources are insufficient', 'Decided to trigger the web crawler to fetch the latest info');

    // 4. Crawling
    await new Promise((resolve) => setTimeout(resolve, 1500));
    addEvent('crawling', 'Searching the web and downloading the latest papers...', 'Target: arXiv:2401.xxxxx.pdf');

    // 5. OCR
    await new Promise((resolve) => setTimeout(resolve, 1500));
    addEvent('ocr', 'Extracting text via OCR...', 'Parsing PDF page 1/12, extracting 3 paragraphs');

    // 6. Generating
    await new Promise((resolve) => setTimeout(resolve, 1200));
    addEvent('generating', 'Generating the final answer...', 'Synthesizing retrieved and crawled content');

    // Simulated latency (ms -> s) so the dashboard has a visible trend
    const retrievalTime = 0.5 + Math.random() * 0.4; // 0.5 ~ 0.9s
    const generationTime = 1.0 + Math.random() * 0.8; // 1.0 ~ 1.8s

    const { answer, sources, confidence } = buildMockAnswer(task.query);

    task.status = 'completed';
    task.result = {
      answer,
      sources,
      confidence,
      retrievalTime,
      generationTime,
    };
    this.tasks.set(id, task);
  }
}

// Global singleton pattern to prevent data loss during hot reload in development
const globalForStore = global as unknown as { taskStore: TaskStore };

export const taskStore = globalForStore.taskStore || new TaskStore();

if (process.env.NODE_ENV !== 'production') globalForStore.taskStore = taskStore;
