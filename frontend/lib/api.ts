import type { FaceSegmentationResponse } from "@/types/faceSegmentation";
import type { FaceMeshTopology, FacialLandmarkResponse } from "@/types/facialLandmarks";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(`Cannot reach backend at ${API_BASE}. Is it running?`);
  }
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<T>;
}

export function detectLandmarks(file: File): Promise<FacialLandmarkResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/facial-analysis/landmarks", { method: "POST", body: form });
}

export function fetchTopology(): Promise<FaceMeshTopology> {
  return request("/api/facial-analysis/topology");
}

export function segmentFace(file: File): Promise<FaceSegmentationResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/facial-analysis/segmentation", { method: "POST", body: form });
}
