"use client";

import { useEffect, useRef, useState } from "react";
import type { Overlay } from "@/lib/segmentation";
import type { Connection, FaceMeshTopology, FacialLandmarkResponse, Landmark } from "@/types/facialLandmarks";

export interface ViewerLayers {
  /** Off = draw the landmarks on a plain dark background instead of the photo. */
  image: boolean;
  points: boolean;
  mesh: boolean;
  contours: boolean;
  measurements: boolean;
}

interface Props {
  imageUrl: string;
  result: FacialLandmarkResponse | null;
  topology: FaceMeshTopology | null;
  layers: ViewerLayers;
  /** Optional segmentation overlay (image-pixel coordinates). Absent = viewer behaves as before. */
  overlay?: Overlay | null;
  /** 0..1, applied at draw time so changing it never recomposes the overlay. */
  overlayOpacity?: number;
  /** 0..1 position of the wipe divider: the overlay shows left of it, the plain image right of it. Null = no wipe. */
  compare?: number | null;
  /** Text describing an image pixel (region/zones), shown in the hover tooltip. */
  describePixel?: (imgX: number, imgY: number) => string | null;
}

const CONTOUR_COLORS: Record<string, string> = {
  face_oval: "#ffffff",
  lips: "#ff4d6d",
  left_eye: "#4dabf7",
  right_eye: "#4dabf7",
  left_eyebrow: "#ffd43b",
  right_eyebrow: "#ffd43b",
  left_iris: "#e599f7",
  right_iris: "#e599f7",
  nose: "#69db7c",
};

// Landmarks used by backend geometry.py, highlighted when "measurements" is on.
const KEY_POINTS = [33, 133, 263, 362, 468, 473, 61, 291];

/**
 * Draws the image on a canvas at its natural resolution, then overlays
 * landmarks. Normalized coords map to canvas pixels as (x * width, y * height).
 * CSS scales the canvas to fit, so the overlay always stays aligned with
 * the image.
 */
export default function FaceLandmarkViewer({
  imageUrl,
  result,
  topology,
  layers,
  overlay = null,
  overlayOpacity = 0.55,
  compare = null,
  describePixel,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [hover, setHover] = useState<{ lm: Landmark; cssX: number; cssY: number } | null>(null);
  const [pixelInfo, setPixelInfo] = useState<{ text: string; cssX: number; cssY: number } | null>(null);

  useEffect(() => {
    const image = new Image();
    // Modern browsers apply EXIF orientation to naturalWidth/Height and
    // drawImage. The backend does the same, so the coordinates agree.
    image.onload = () => setImg(image);
    image.src = imageUrl;
    return () => {
      image.onload = null;
    };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d")!;
    // The canvas keeps the photo's size even when the photo is hidden, so the
    // landmark positions don't move when you toggle it.
    if (layers.image) {
      ctx.drawImage(img, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, W, H);
    }

    if (overlay) {
      ctx.save();
      if (compare !== null) {
        ctx.beginPath();
        ctx.rect(0, 0, compare * W, H);
        ctx.clip();
      }
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(overlay.canvas, overlay.x, overlay.y);
      ctx.restore();
    }
    if (overlay && compare !== null) {
      ctx.fillStyle = "#fff";
      ctx.fillRect(compare * W - Math.max(W, H) / 800, 0, (2 * Math.max(W, H)) / 800, H);
    }

    if (!result?.face_detected) return;
    const lms = result.landmarks;
    const px = (lm: Landmark): [number, number] => [lm.x * W, lm.y * H];
    // Line and point sizes scale with the image, so they look the same at any resolution.
    const unit = Math.max(W, H) / 800;

    const strokeConnections = (conns: Connection[], color: string, width: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (const [a, b] of conns) {
        if (!lms[a] || !lms[b]) continue;
        const [x1, y1] = px(lms[a]);
        const [x2, y2] = px(lms[b]);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
    };

    if (layers.mesh && topology) {
      strokeConnections(topology.tesselation, "rgba(0, 255, 170, 0.35)", 0.6 * unit);
    }
    if (layers.contours && topology) {
      for (const [name, conns] of Object.entries(topology.contours)) {
        strokeConnections(conns, CONTOUR_COLORS[name] ?? "#fff", 1.5 * unit);
      }
    }
    if (layers.points) {
      ctx.fillStyle = "#00ff88";
      for (const lm of lms) {
        const [x, y] = px(lm);
        ctx.beginPath();
        ctx.arc(x, y, 1.4 * unit, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (layers.measurements && result.measurements) {
      const b = result.measurements.face_bounding_box;
      ctx.strokeStyle = "#ff922b";
      ctx.lineWidth = 2 * unit;
      ctx.strokeRect(b.x_min, b.y_min, b.width, b.height);
      ctx.fillStyle = "#ff0000";
      for (const i of KEY_POINTS) {
        if (!lms[i]) continue;
        const [x, y] = px(lms[i]);
        ctx.beginPath();
        ctx.arc(x, y, 3 * unit, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (hover) {
      const [x, y] = px(hover.lm);
      ctx.strokeStyle = "#ff0";
      ctx.lineWidth = 2 * unit;
      ctx.beginPath();
      ctx.arc(x, y, 5 * unit, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [img, result, topology, layers, hover, overlay, overlayOpacity, compare]);

  // Hover shows the nearest landmark's index, so you can check that e.g.
  // index 1 is the nose tip and 152 is the chin.
  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!result?.face_detected || !img) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const text = describePixel?.(nx * img.naturalWidth, ny * img.naturalHeight) ?? null;
    setPixelInfo(text ? { text, cssX: e.clientX - rect.left, cssY: e.clientY - rect.top } : null);
    const aspect = rect.width / rect.height;
    let best: Landmark | null = null;
    let bestD = Infinity;
    for (const lm of result.landmarks) {
      const d = ((lm.x - nx) * aspect) ** 2 + (lm.y - ny) ** 2;
      if (d < bestD) {
        bestD = d;
        best = lm;
      }
    }
    // Only snap when the cursor is close to a point (about 3% of image height).
    if (best && bestD < 0.03 ** 2) {
      setHover({ lm: best, cssX: e.clientX - rect.left, cssY: e.clientY - rect.top });
    } else {
      setHover(null);
    }
  };

  const aspectMismatch =
    img && result && Math.abs(img.naturalWidth / img.naturalHeight - result.image.width / result.image.height) > 0.01;

  return (
    <div>
      {aspectMismatch && (
        <p style={{ color: "#c92a2a" }}>
          Warning: the displayed image ({img!.naturalWidth}×{img!.naturalHeight}) and the backend image (
          {result!.image.width}×{result!.image.height}) have different aspect ratios. Likely an EXIF rotation problem.
        </p>
      )}
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
        <canvas
          ref={canvasRef}
          onMouseMove={onMouseMove}
          onMouseLeave={() => {
            setHover(null);
            setPixelInfo(null);
          }}
          style={{ display: "block", maxWidth: "100%", maxHeight: "80vh", height: "auto", cursor: "crosshair" }}
        />
        {hover && (
          <div
            style={{
              position: "absolute",
              left: hover.cssX + 12,
              top: hover.cssY + 12,
              background: "rgba(0,0,0,0.8)",
              color: "#fff",
              font: "12px ui-monospace, monospace",
              padding: "4px 6px",
              borderRadius: 4,
              pointerEvents: "none",
              whiteSpace: "pre",
            }}
          >
            {`#${hover.lm.index}\nx ${hover.lm.x.toFixed(4)}\ny ${hover.lm.y.toFixed(4)}\nz ${hover.lm.z.toFixed(4)}` +
              (pixelInfo ? `\n${pixelInfo.text}` : "")}
          </div>
        )}
        {!hover && pixelInfo && (
          <div
            style={{
              position: "absolute",
              left: pixelInfo.cssX + 12,
              top: pixelInfo.cssY + 12,
              background: "rgba(0,0,0,0.8)",
              color: "#fff",
              font: "12px ui-monospace, monospace",
              padding: "4px 6px",
              borderRadius: 4,
              pointerEvents: "none",
              whiteSpace: "pre",
            }}
          >
            {pixelInfo.text}
          </div>
        )}
      </div>
    </div>
  );
}
