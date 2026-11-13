// Client for POST /api/makeup/preview. Never logs request/response bodies or images.
import { API_BASE } from "@/lib/api";
import type { ComponentRenderReport, PreviewResponse, SkipReason } from "@/types/makeupPreview";

const MAX_RETRY_SECONDS = 60;

export class PreviewApiError extends Error {
  readonly kind: string;
  readonly status?: number;
  constructor(kind: string, message: string, status?: number) {
    super(message);
    this.name = "PreviewApiError";
    this.kind = kind;
    this.status = status;
  }
}

export class PreviewRetryError extends PreviewApiError {
  readonly retryAfterSeconds: number | null;
  constructor(kind: string, message: string, status: number, retryAfterSeconds: number | null) {
    super(kind, message, status);
    this.name = "PreviewRetryError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function retryAfterSeconds(res: Response): number | null {
  const raw = res.headers.get("Retry-After")?.trim();
  if (raw === undefined || !/^[0-9]+$/.test(raw)) return null;
  const digits = raw.replace(/^0+/, "") || "0";
  if (digits.length > 2) return MAX_RETRY_SECONDS;
  return Math.min(MAX_RETRY_SECONDS, Math.max(0, Number(digits)));
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as Record<string, unknown>).detail;
      return typeof detail === "string" ? detail : "The request was not valid.";
    }
  } catch {
    // fall through to status text
  }
  return `${res.status} ${res.statusText}`;
}

function isPreviewResponse(body: unknown): body is PreviewResponse {
  if (typeof body !== "object" || body === null) return false;
  const b = body as Record<string, unknown>;
  return (
    b.schema_version === "1.0" &&
    typeof b.image_png_base64 === "string" &&
    typeof b.width === "number" &&
    Number.isInteger(b.width) &&
    b.width > 0 &&
    typeof b.height === "number" &&
    Number.isInteger(b.height) &&
    b.height > 0 &&
    typeof b.plan_status === "string" &&
    Array.isArray(b.reports) &&
    Array.isArray(b.warnings)
  );
}

export async function requestPreview(selfie: File, reference: File, signal?: AbortSignal): Promise<PreviewResponse> {
  const form = new FormData();
  form.append("selfie", selfie);
  form.append("reference", reference);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/makeup/preview`, { method: "POST", body: form, signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new PreviewApiError("network", `Cannot reach backend at ${API_BASE}. Is it running?`);
  }

  if (!res.ok) {
    const detail = await errorDetail(res);
    if (res.status === 429 || res.status === 503) {
      throw new PreviewRetryError(res.status === 429 ? "rate_limited" : "busy", detail, res.status, retryAfterSeconds(res));
    }
    throw new PreviewApiError("http", detail, res.status);
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new PreviewApiError("invalid_response", "The server returned an unreadable response.");
  }
  if (!isPreviewResponse(body)) {
    throw new PreviewApiError("invalid_response", "The server returned an unexpected response.");
  }
  return body;
}

const SKIP_TEXT: Record<SkipReason, string> = {
  no_mask: "No region found for this feature",
  mask_invalid: "Region mask was unusable",
  no_colour: "No reference colour available",
  zero_opacity: "Intensity too low to show",
  empty_mask: "Region was empty on this face",
};

export function skipReasonText(r: ComponentRenderReport): string {
  return r.reason ? (SKIP_TEXT[r.reason] ?? r.reason) : "Skipped";
}

export function errorMessage(e: unknown): string {
  if (e instanceof PreviewRetryError) {
    const n = e.retryAfterSeconds;
    if (e.status === 429) {
      return n !== null ? `Too many requests, try again in ${n} s` : "Too many requests, please wait a moment and try again.";
    }
    if (e.status === 503) {
      return `The server is busy, try again shortly${n !== null ? ` (${n} s)` : ""}`;
    }
  }
  return e instanceof Error ? e.message : String(e);
}
