import { type NextRequest, NextResponse } from "next/server"
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;

// Vector retrieval endpoint
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { questionId, query } = body

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding retrieve to: ${BACKEND_URL}/api/v1/chat/retrieve`);
        const response = await axios.post(`${BACKEND_URL}/api/v1/chat/retrieve`, body);
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend retrieve failed, falling back to mock:', error.message);
      }
    }

    console.log("[v0] Retrieval endpoint received a request (Mock):", { questionId, query })

    // Simulated vector retrieval delay
    await new Promise((resolve) => setTimeout(resolve, 800))

    // Mock retrieval results
    const mockResults = [
      {
        id: "doc_1",
        content: "Multimodal AI can process text, images, and audio simultaneously, enabling smarter interactions.",
        similarity: 0.92,
        source: "Knowledge base doc A",
      },
      {
        id: "doc_2",
        content: "Vector retrieval converts text into high-dimensional vectors and computes similarity in the vector space to find the most relevant content.",
        similarity: 0.87,
        source: "Knowledge base doc B",
      },
      {
        id: "doc_3",
        content: "Self-improvement allows AI systems to learn from user interactions and continuously optimize answer quality.",
        similarity: 0.85,
        source: "Knowledge base doc C",
      },
    ]

    const response = {
      success: true,
      data: {
        questionId,
        results: mockResults,
        totalResults: mockResults.length,
        retrievalTime: 0.8,
      },
      message: "Retrieval complete",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Retrieval endpoint error:", error)
    return NextResponse.json({ success: false, message: "Retrieval failed" }, { status: 500 })
  }
}
