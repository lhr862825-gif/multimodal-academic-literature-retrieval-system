import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL;

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file');

    if (!file) {
      return NextResponse.json(
        { success: false, message: 'File is required' },
        { status: 400 }
      );
    }

    // 1. Try the real backend first
    if (BACKEND_URL) {
      try {
        console.log(`[Proxy] Forwarding OCR to: ${BACKEND_URL}/api/v1/ocr`);
        const backendData = new FormData();
        backendData.append('file', file);

        const response = await axios.post(`${BACKEND_URL}/api/v1/ocr`, backendData);
        return NextResponse.json(response.data);
      } catch (error: any) {
        console.warn('[Proxy] Backend OCR failed, falling back to mock:', error.message);
      }
    }

    // Mock OCR processing
    console.log("[v0] OCR endpoint received a request (Mock)");
    await new Promise((resolve) => setTimeout(resolve, 2000));

    return NextResponse.json({
      success: true,
      data: {
        text: "Mock OCR Result: This is the text extracted from the image.",
        confidence: 0.98,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, message: 'OCR Processing Failed' },
      { status: 500 }
    );
  }
}
