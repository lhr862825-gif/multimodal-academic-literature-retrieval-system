import { NextRequest, NextResponse } from 'next/server';
import { taskStore } from '@/lib/store';
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;
const DEMO_MODE = process.env.DEMO_MODE === 'true';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  // Demo mode: return the built-in mock engine's task status and timeline
  if (DEMO_MODE || !BACKEND_URL) {
    const task = taskStore.getTask(id);
    if (!task) {
      return NextResponse.json(
        { success: false, message: 'Task not found' },
        { status: 404 }
      );
    }
    return NextResponse.json({
      success: true,
      data: {
        status: task.status,
        timeline: (task as any).timeline,
        result: task.result,
        error: task.error,
      },
    });
  }

  // Real backend mode
  try {
    // Adapter for the real backend endpoint: /api/task/{id}
    const response = await axios.get(`${BACKEND_URL}/api/task/${id}`);

    // Map the backend response into the frontend store format
    return NextResponse.json({
      success: true,
      data: response.data,
    });
  } catch (error: any) {
    console.error('Backend error:', error.message);
    return NextResponse.json(
      { success: false, message: error.response?.data?.detail || error.message || 'Backend service error' },
      { status: error.response?.status || 500 }
    );
  }
}
