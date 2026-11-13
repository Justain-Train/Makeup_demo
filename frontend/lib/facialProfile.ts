// FacialProfile API client. Never logs response bodies or facial geometry.

import { API_BASE } from "@/lib/api";
import type { AvailableComponent, Component, FacialProfile, FeaturesData } from "@/types/facialProfile";

export function isComponentAvailable<T>(c: Component<T>): c is AvailableComponent<T> {
  return c.meta.status !== "unavailable";
}

export function getFeatures(profile: FacialProfile): FeaturesData | null {
  const component = profile.components?.features;
  if (!component || !isComponentAvailable(component)) return null;
  return component.data;
}

/** Mirrors backend/app/services/facial_profile/compat.py. MAJOR -> minimum supported MINOR. */
export const SUPPORTED_PROFILE_MAJORS: ReadonlyMap<number, number> = new Map([[1, 0]]);

const VERSION_RE = /^[0-9]{1,4}\.[0-9]{1,4}$/;
const MAX_VERSION_LEN = 16;
const DEFAULT_RETRY_SECONDS = 2;
const MAX_RETRY_SECONDS = 60;

export type ProfileApiErrorKind = "network" | "http" | "unsupported_version" | "invalid_response";

export class ProfileApiError extends Error {
  readonly kind: ProfileApiErrorKind;
  readonly status?: number;

  constructor(kind: ProfileApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ProfileApiError";
    this.kind = kind;
    this.status = status;
  }
}

export class ProfileBusyError extends ProfileApiError {
  readonly retryAfterSeconds: number;

  constructor(message: string, retryAfterSeconds: number) {
    super("http", message, 503);
    this.name = "ProfileBusyError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export class UnsupportedProfileVersionError extends ProfileApiError {
  constructor(version?: unknown) {
    const shown = typeof version === "string" ? version.slice(0, MAX_VERSION_LEN) : "unknown";
    super("unsupported_version", `Unsupported profile schema version: ${shown}`);
    this.name = "UnsupportedProfileVersionError";
  }
}

export function parseProfileVersion(s: string): [number, number] {
  if (typeof s !== "string" || s.length > MAX_VERSION_LEN || !VERSION_RE.test(s)) {
    throw new Error("Invalid schema version; expected 'MAJOR.MINOR'");
  }
  const [major, minor] = s.split(".");
  return [Number(major), Number(minor)];
}

export function isSupportedProfileVersion(version: string): boolean {
  const [major, minor] = parseProfileVersion(version);
  const minMinor = SUPPORTED_PROFILE_MAJORS.get(major);
  return minMinor !== undefined && minor >= minMinor;
}

export function checkProfileVersion(body: unknown): asserts body is { schema_version: string } {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw new UnsupportedProfileVersionError();
  }
  const version = (body as Record<string, unknown>).schema_version;
  if (typeof version !== "string") throw new UnsupportedProfileVersionError();
  let supported: boolean;
  try {
    supported = isSupportedProfileVersion(version);
  } catch {
    throw new UnsupportedProfileVersionError(version);
  }
  if (!supported) throw new UnsupportedProfileVersionError(version);
}

function retryAfterSeconds(res: Response): number {
  const raw = res.headers.get("Retry-After")?.trim();
  if (raw === undefined || !/^[0-9]+$/.test(raw)) return DEFAULT_RETRY_SECONDS;
  const digits = raw.replace(/^0+/, "") || "0";
  if (digits.length > 2) return MAX_RETRY_SECONDS;
  return Math.min(MAX_RETRY_SECONDS, Math.max(0, Number(digits)));
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (typeof body === "object" && body !== null) {
      const detail = (body as Record<string, unknown>).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    // fall through to status text
  }
  return `${res.status} ${res.statusText}`;
}

export async function fetchFacialProfile(
  file: File,
  opts?: { includeMasks?: boolean; signal?: AbortSignal },
): Promise<FacialProfile> {
  const form = new FormData();
  form.append("file", file);
  const query = opts?.includeMasks === false ? "?include_masks=false" : "";

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/facial-analysis/profile${query}`, {
      method: "POST",
      body: form,
      signal: opts?.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ProfileApiError("network", `Cannot reach backend at ${API_BASE}. Is it running?`);
  }

  if (res.status === 503) {
    throw new ProfileBusyError(await errorDetail(res), retryAfterSeconds(res));
  }
  if (!res.ok) {
    throw new ProfileApiError("http", await errorDetail(res), res.status);
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new ProfileApiError("invalid_response", "The server returned an invalid response.", res.status);
  }
  checkProfileVersion(body);
  return body as FacialProfile;
}
