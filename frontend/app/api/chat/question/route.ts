import { type NextRequest, NextResponse } from "next/server"
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;

// User question submission endpoint
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { message, conversationId, attachments } = body

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding question to: ${BACKEND_URL}/api/v1/chat/question`);
        const response = await axios.post(`${BACKEND_URL}/api/v1/chat/question`, body);
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend question ack failed, falling back to mock:', error.message);
      }
    }

    console.log("[v0] Question endpoint received a request (Mock):", { message, conversationId, attachments })

    // Simulated processing delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock response data
    const response = {
      success: true,
      data: {
        questionId: `q_${Date.now()}`,
        conversationId: conversationId || `conv_${Date.now()}`,
        message,
        timestamp: new Date().toISOString(),
        attachments: attachments || [],
      },
      message: "Question received",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Question endpoint error:", error)
    return NextResponse.json({ success: false, message: "Processing failed" }, { status: 500 })
  }
}
