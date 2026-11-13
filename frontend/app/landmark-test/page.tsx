"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import FaceLandmarkViewer, { type ViewerLayers } from "@/components/FaceLandmarkViewer";
import SegmentationControls, { type SegmentationSelection } from "@/components/SegmentationControls";
import { detectLandmarks, fetchTopology, segmentFace } from "@/lib/api";
import { composeOverlay, decodeSegmentation, describePixel, type DecodedSegmentation } from "@/lib/segmentation";
import type { FaceSegmentationResponse } from "@/types/faceSegmentation";
import type { FaceMeshTopology, FacialLandmarkResponse } from "@/types/facialLandmarks";

// Debug-only page: upload a selfie, see what MediaPipe returns.
export default function LandmarkTestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<FacialLandmarkResponse | null>(null);
  const [topology, setTopology] = useState<FaceMeshTopology | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [seg, setSeg] = useState<FaceSegmentationResponse | null>(null);
  const [decoded, setDecoded] = useState<DecodedSegmentation | null>(null);
  const [selection, setSelection] = useState<SegmentationSelection>({ regions: new Set(), zones: new Set() });
  const [opacity, setOpacity] = useState(0.55);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compare, setCompare] = useState(0.5);
  const [layers, setLayers] = useState<ViewerLayers>({ image: true, points: true, mesh: false, contours: true, measurements: false });

  // Mesh topology is static, so fetch it once. If it fails, points still work.
  useEffect(() => {
    fetchTopology().then(setTopology).catch(() => setTopology(null));
  }, []);

  // Decode the segmentation PNGs whenever a new response arrives.
  useEffect(() => {
    setDecoded(null);
    if (!seg || seg.status !== "ok") return;
    let cancelled = false;
    decodeSegmentation(seg)
      .then((d) => {
        if (!cancelled) setDecoded(d);
      })
      .catch((err) => {
        if (!cancelled) setError(`Could not decode segmentation masks: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [seg]);

  // The overlay is recomposed only when the selection (or data) changes;
  // opacity and the compare divider are applied at draw time.
  const overlay = useMemo(
    () => (seg && decoded ? composeOverlay(seg, decoded, selection) : null),
    [seg, decoded, selection],
  );

  // Release the old object URL whenever the image changes.
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setSeg(null);
    setSelection({ regions: new Set(), zones: new Set() });
    setCompareEnabled(false);
    setCompare(0.5);
    setError(null);
    setImageUrl(f ? URL.createObjectURL(f) : null);
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      setSeg(null);
      setResult(await detectLandmarks(file));
      setElapsedMs(performance.now() - t0);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const onSegment = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      const r = await segmentFace(file);
      setSeg(r);
      setResult(r.landmarks); // the same coordinate space, so landmarks line up with the masks
      // Start with the face-skin region and the key zones visible.
      setSelection({
        regions: new Set(),
        zones: new Set(r.zones.filter((z) => z.mask && ["face_skin"].includes(z.id)).map((z) => z.id)),
      });
      setElapsedMs(performance.now() - t0);
    } catch (err) {
      setSeg(null);
      setResult(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "landmarks.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <main style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
      <h1 style={{ fontSize: 22 }}>Facial landmark test</h1>
      <p>
        <Link href="/preview">Makeup preview →</Link>
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <input type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
        <button onClick={onAnalyze} disabled={!file || loading}>
          {loading ? "Analyzing…" : "Detect landmarks"}
        </button>
        <button onClick={onSegment} disabled={!file || loading}>
          {loading ? "Analyzing…" : "Segment face"}
        </button>
        {(Object.keys(layers) as (keyof ViewerLayers)[]).map((k) => (
          <label key={k} style={{ fontSize: 14 }}>
            <input
              type="checkbox"
              checked={layers[k]}
              disabled={(k === "mesh" || k === "contours") && !topology}
              onChange={() => setLayers((l) => ({ ...l, [k]: !l[k] }))}
            />{" "}
            {k}
          </label>
        ))}
      </div>

      {!topology && <p style={{ color: "#888", fontSize: 13 }}>Mesh topology not loaded (is the backend running?).</p>}
      {error && <p style={{ color: "#c92a2a" }}>Error: {error}</p>}
      {/* Any non-ok status (no face, multiple faces, ...) shows the backend's message. */}
      {result && result.status !== "ok" && (
        <p style={{ color: "#e67700" }} role="alert">
          {result.message ?? result.status}
        </p>
      )}

      {seg && seg.status !== "ok" && (
        <p style={{ color: "#555", fontSize: 12 }}>License: {seg.model.license_warning}</p>
      )}

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
        {imageUrl && (
          <FaceLandmarkViewer
            imageUrl={imageUrl}
            result={result}
            topology={topology}
            layers={layers}
            overlay={overlay}
            overlayOpacity={opacity}
            compare={overlay && compareEnabled ? compare : null}
            describePixel={seg && decoded ? (x, y) => describePixel(seg, decoded, x, y) : undefined}
          />
        )}

        {seg?.status === "ok" && (
          <SegmentationControls
            seg={seg}
            selection={selection}
            onSelectionChange={setSelection}
            opacity={opacity}
            onOpacityChange={setOpacity}
            compareEnabled={compareEnabled}
            onCompareEnabledChange={setCompareEnabled}
            compare={compare}
            onCompareChange={setCompare}
          />
        )}

        {result?.status === "ok" && (
          <aside style={{ fontSize: 13, minWidth: 260 }}>
            <p>
              <b>{result.landmark_count}</b> landmarks · {result.image.width}×{result.image.height}px
              {result.face_count > 1 && <> · {result.face_count} faces found (largest analyzed)</>}
              {elapsedMs !== null && <> · {Math.round(elapsedMs)} ms round-trip</>}
            </p>
            {result.measurements && (
              <table style={{ borderCollapse: "collapse" }}>
                <tbody>
                  {Object.entries(result.measurements)
                    .filter(([, v]) => typeof v === "number")
                    .map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ paddingRight: 12 }}>{k}</td>
                        <td style={{ fontFamily: "monospace", textAlign: "right" }}>{(v as number).toFixed(1)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
            <p style={{ color: "#666" }}>Hover over the image to see the nearest landmark index.</p>
            <button onClick={downloadJson}>Download JSON</button>
          </aside>
        )}
      </div>
    </main>
  );
}
