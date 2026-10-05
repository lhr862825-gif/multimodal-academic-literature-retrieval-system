import { type NextRequest, NextResponse } from "next/server"
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;

// Crawler trigger endpoint
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { url, depth = 1, maxPages = 10 } = body

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding crawler trigger to: ${BACKEND_URL}/api/v1/crawler/trigger`);
        const response = await axios.post(`${BACKEND_URL}/api/v1/crawler/trigger`, body);
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend crawler trigger failed, falling back to mock:', error.message);
      }
    }

    console.log("[v0] Crawler trigger endpoint received a request (Mock):", { url, depth, maxPages })

    // Simulated crawler startup delay
    await new Promise((resolve) => setTimeout(resolve, 600))

    // Mock crawler task data
    const taskId = `crawler_${Date.now()}`

    const response = {
      success: true,
      data: {
        taskId,
        url,
        status: "running",
        config: {
          depth,
          maxPages,
        },
        startTime: new Date().toISOString(),
        estimatedTime: 30, // seconds
      },
      message: "Crawler task started",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Crawler trigger endpoint error:", error)
    return NextResponse.json({ success: false, message: "Failed to start" }, { status: 500 })
  }
}

// Query crawler task status
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const taskId = searchParams.get("taskId")

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding crawler status check to: ${BACKEND_URL}/api/v1/crawler/status`);
        const response = await axios.get(`${BACKEND_URL}/api/v1/crawler/status`, { params: { taskId } });
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend crawler status check failed, falling back to mock:', error.message);
      }
    }

    console.log("[v0] Querying crawler status (Mock):", taskId)

    // Mock crawler status data
    const response = {
      success: true,
      data: {
        taskId,
        status: "completed",
        progress: 100,
        pagesScraped: 8,
        documentsCreated: 15,
        completedTime: new Date().toISOString(),
        results: [
          { url: "https://example.com/page1", title: "Page 1", status: "success" },
          { url: "https://example.com/page2", title: "Page 2", status: "success" },
          { url: "https://example.com/page3", title: "Page 3", status: "success" },
        ],
      },
      message: "Task completed",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Query crawler status error:", error)
    return NextResponse.json({ success: false, message: "Query failed" }, { status: 500 })
  }
}
