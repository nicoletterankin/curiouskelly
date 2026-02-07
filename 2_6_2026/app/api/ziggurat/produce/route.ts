import { NextResponse } from "next/server";
import { start } from "workflow/api";
import { produceZigguratVideo } from "@/workflows/ziggurat-production";

export async function POST(request: Request) {
  try {
    const { kellyImageUrl } = await request.json();

    if (!kellyImageUrl) {
      return NextResponse.json({ error: "kellyImageUrl is required" }, { status: 400 });
    }

    const workflowId = await start(produceZigguratVideo, [kellyImageUrl]);

    return NextResponse.json(
      {
        success: true,
        workflowId,
        message: "Video production started. Monitor progress with: npx workflow web",
      },
      { status: 202 }
    );
  } catch (error) {
    console.error("Production error:", error);
    return NextResponse.json(
      { error: "Failed to start production", details: String(error) },
      { status: 500 }
    );
  }
}
