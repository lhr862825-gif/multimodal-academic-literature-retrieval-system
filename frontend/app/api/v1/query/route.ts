import { NextRequest, NextResponse } from 'next/server';
import { taskStore } from '@/lib/store';
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;
const DEMO_MODE = process.env.DEMO_MODE === 'true';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { query } = body;

    if (!query) {
      return NextResponse.json(
        { success: false, message: 'Query is required' },
        { status: 400 }
      );
    }

    // Demo mode: no real backend, use the built-in mock engine for the full agent pipeline
    if (DEMO_MODE || !BACKEND_URL) {
      console.log(`[Demo] Handling question with the built-in mock engine: ${query}`);
      const task = taskStore.createTask(query);
      return NextResponse.json({
        success: true,
        data: {
          taskId: task.id,
          message: 'Task submitted, processing...',
        },
      });
    }

    // Real backend mode
    try {
      console.log(`Forwarding query to backend: ${BACKEND_URL}/api/task/create`);
      // Adapter for the real backend endpoint: /api/task/create
      // Payload format: { "question": "..." }
      const payload = {
        question: query,
      };

      const response = await axios.post(`${BACKEND_URL}/api/task/create`, payload);

      // Map the response format: { "task_id": "...", "message": "..." } -> { success: true, data: { taskId: ... } }
      return NextResponse.json({
        success: true,
        data: {
          taskId: response.data.task_id,
          message: response.data.message,
        },
      });
    } catch (error: any) {
      console.error('Backend error:', error.message);
      return NextResponse.json(
        { success: false, message: error.response?.data?.detail || error.message || 'Backend service error' },
        { status: error.response?.status || 500 }
      );
    }
  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { success: false, message: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
