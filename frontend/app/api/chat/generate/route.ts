import { type NextRequest, NextResponse } from "next/server"
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;

// Answer generation endpoint
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { questionId, query, retrievedDocs, conversationHistory, chainLength } = body

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding generate to: ${BACKEND_URL}/api/v1/chat/generate`);
        const response = await axios.post(`${BACKEND_URL}/api/v1/chat/generate`, body);
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend generate failed, falling back to mock:', error.message);
        // If the backend call fails (e.g. endpoint not implemented), fall back to the mock logic below
      }
    }

    console.log("[v0] Generation endpoint received a request (Mock):", { questionId, query })

    // Simulated AI generation delay
    await new Promise((resolve) => setTimeout(resolve, 1500))

    // Generate a different mock answer based on the question content
    let answer = ""
    const lowerQuery = query.toLowerCase()

    if (lowerQuery.includes("multimodal")) {
      answer =
        "Multimodal AI is a technology that can simultaneously process and understand multiple data types (text, images, audio, video, etc.). By fusing information from different modalities, it achieves more comprehensive and intelligent understanding and interaction.\n\nKey features include:\n1. Cross-modal understanding: recognizing the relationships between different data types\n2. Unified representation: mapping different modalities into a shared feature space\n3. Synergistic enhancement: multiple modalities complement each other to improve overall performance\n\nIt has broad applications including intelligent customer service, content creation, and medical diagnosis."
    } else if (lowerQuery.includes("vector") || lowerQuery.includes("retrieval")) {
      answer =
        "Vector retrieval is an information retrieval technique based on semantic similarity. It works as follows:\n\n1. Text embedding: a pre-trained model converts text into high-dimensional vectors\n2. Similarity scoring: the similarity between the query vector and document vectors is computed in the vector space\n3. Ranking: the most relevant documents are returned by similarity score\n\nCompared with traditional keyword search, vector retrieval understands semantics and finds content that is phrased differently but means the same thing, producing better retrieval results."
    } else if (lowerQuery.includes("learning") || lowerQuery.includes("self")) {
      answer =
        "Self-improvement refers to an AI system's ability to continuously learn and optimize from user interactions and feedback.\n\nCore mechanisms include:\n1. Feedback collection: recording user satisfaction and corrections\n2. Model fine-tuning: continuously adjusting model parameters with new data\n3. Knowledge updates: automatically expanding and refreshing the knowledge base\n4. Quality monitoring: evaluating answer quality in real time and optimizing accordingly\n\nThis allows the AI system to get smarter with use and better adapt to specific scenarios and user needs."
    } else {
      answer = `I understand your question is about "${query}". Based on my knowledge-base retrieval, here is the information:\n\nThis is a multimodal AI assistant demo system that supports text, images, files, and voice inputs. The system uses vector retrieval to quickly locate relevant knowledge and a self-improvement mechanism to continuously optimize answer quality.\n\nIf you have a more specific question, feel free to keep asking!`
    }

    const chainLen = typeof chainLength === "number" ? chainLength : 3
    const baseSteps = [
      "Identify question type",
      "Retrieve relevant documents",
      "Filter and rank",
    ]
    const extraSteps =
      chainLen >= 10
        ? ["Extract key points", "Draft answer", "Consistency check", "Reflect and revise"]
        : chainLen >= 6
        ? ["Extract key points", "Draft answer"]
        : chainLen >= 4
        ? ["Extract key points"]
        : []
    const thinking = [...baseSteps, ...extraSteps].map((s, i) => `${i + 1}. ${s}`)
    const response = {
      success: true,
      data: {
        questionId,
        answer,
        confidence: 0.89,
        sources: retrievedDocs?.map((doc: any) => doc.source) || ["Knowledge base"],
        generationTime: 1.5,
        tokens: {
          prompt: 256,
          completion: 128,
          total: 384,
        },
        thinking,
      },
      message: "Generation complete",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Generation endpoint error:", error)
    return NextResponse.json({ success: false, message: "Generation failed" }, { status: 500 })
  }
}
