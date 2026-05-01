import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// process.cwd() returns /workspace/frontend inside the container (the mounted
// host volume), so the log persists on the host at ./frontend/frontend_runtime_errors.log
const LOG_PATH = path.join(process.cwd(), 'frontend_runtime_errors.log');

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const body = (await request.json()) as Record<string, unknown>;
    const entry = JSON.stringify({ timestamp: new Date().toISOString(), ...body }) + '\n';
    fs.appendFileSync(LOG_PATH, entry);
    return NextResponse.json({ ok: true, path: LOG_PATH }, { status: 200 });
  } catch (err) {
    console.error('[/api/log] Write failed:', err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function DELETE(): Promise<NextResponse> {
  try {
    fs.writeFileSync(LOG_PATH, '');
    return NextResponse.json({ ok: true, truncated: true }, { status: 200 });
  } catch (err) {
    console.error('[/api/log] Truncate failed:', err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
