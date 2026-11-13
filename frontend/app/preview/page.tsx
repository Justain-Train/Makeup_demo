"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import { errorMessage, requestPreview, skipReasonText } from "@/lib/makeupPreview";
import type { PreviewResponse } from "@/types/makeupPreview";

const ACCEPT = "image/jpeg,image/png,image/webp";

function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

export default function PreviewPage() {
  const [selfie, setSelfie] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const selfieUrl = useObjectUrl(selfie);
  const referenceUrl = useObjectUrl(reference);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PreviewResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const onGenerate = async () => {
    if (!selfie || !reference) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const r = await requestPreview(selfie, reference, controller.signal);
      if (controller.signal.aborted) return;
      setResult(r);
    } catch (e) {
      if (controller.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
      setError(errorMessage(e));
    } finally {
      if (abortRef.current === controller) setLoading(false);
    }
  };

  const thumb = (label: string, url: string | null) =>
    url && (
      <figure style={{ margin: 0 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url} alt={label} style={{ maxHeight: 160, maxWidth: 160, display: "block" }} />
        <figcaption style={{ fontSize: 12, color: "#555" }}>{label}</figcaption>
      </figure>
    );

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 22 }}>Makeup preview</h1>
      <p>
        <Link href="/landmark-test">← Back to landmark test</Link>
      </p>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 12 }}>
        <div>
          <label style={{ display: "block", fontSize: 14, marginBottom: 4 }}>
            Selfie
            <br />
            <input
              type="file"
              accept={ACCEPT}
              onChange={(e) => {
                setSelfie(e.target.files?.[0] ?? null);
                setResult(null);
                setError(null);
              }}
            />
          </label>
          {thumb("Selfie", selfieUrl)}
        </div>
        <div>
          <label style={{ display: "block", fontSize: 14, marginBottom: 4 }}>
            Reference look
            <br />
            <input
              type="file"
              accept={ACCEPT}
              onChange={(e) => {
                setReference(e.target.files?.[0] ?? null);
                setResult(null);
                setError(null);
              }}
            />
          </label>
          {thumb("Reference", referenceUrl)}
        </div>
      </div>

      <button onClick={onGenerate} disabled={!selfie || !reference || loading}>
        {loading ? "Generating…" : "Generate preview"}
      </button>

      {loading && (
        <p role="status" style={{ color: "#555" }}>
          Generating preview, this may take a moment…
        </p>
      )}
      {error && (
        <p role="alert" style={{ color: "#c92a2a" }}>
          {error}
        </p>
      )}

      {result && selfieUrl && (
        <section style={{ marginTop: 16 }}>
          <BeforeAfterSlider
            beforeSrc={selfieUrl}
            afterSrc={`data:image/png;base64,${result.image_png_base64}`}
            width={result.width}
            height={result.height}
          />
          <p style={{ fontSize: 14 }}>
            Plan status: <b>{result.plan_status}</b>
          </p>
          {result.warnings.length > 0 && (
            <ul style={{ color: "#e67700", fontSize: 13 }}>
              {result.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
          <ul style={{ fontSize: 13, paddingLeft: 18 }}>
            {result.reports.map((r) => (
              <li key={r.index}>
                #{r.index} {r.status}
                {r.status === "skipped" && ` (${skipReasonText(r)})`} · {r.finish_applied} · {Math.round(r.opacity * 100)}%
                {r.directional_stretch === "deferred" && " · direction: deferred"}
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
