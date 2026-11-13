"use client";

import { useRef, useState } from "react";

interface Props {
  beforeSrc: string;
  afterSrc: string;
  width: number;
  height: number;
}

const imgStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  userSelect: "none",
};

const labelStyle: React.CSSProperties = {
  position: "absolute",
  top: 8,
  padding: "2px 8px",
  background: "rgba(0,0,0,0.6)",
  color: "#fff",
  fontSize: 12,
  borderRadius: 4,
  pointerEvents: "none",
};

export default function BeforeAfterSlider({ beforeSrc, afterSrc, width, height }: Props) {
  const [pos, setPos] = useState(50);
  const [sideBySide, setSideBySide] = useState(false);
  const [mismatch, setMismatch] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromPointer = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };

  const onBeforeLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    if (!naturalWidth || !naturalHeight) return;
    const ratio = naturalWidth / naturalHeight / (width / height);
    setMismatch(Math.abs(ratio - 1) > 0.01);
  };

  const aspectRatio = width / height;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      <div style={{ marginBottom: 8 }}>
        <button type="button" aria-pressed={sideBySide} onClick={() => setSideBySide((v) => !v)}>
          Side by side
        </button>
      </div>
      {mismatch && (
        <p style={{ color: "#e67700", fontSize: 13 }}>
          Original appears rotated or resized differently from the preview; alignment may be off.
        </p>
      )}
      {sideBySide ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {[
            { src: beforeSrc, label: "Before", onLoad: onBeforeLoad },
            { src: afterSrc, label: "After", onLoad: undefined },
          ].map((p) => (
            <div key={p.label} style={{ position: "relative", aspectRatio, background: "#111" }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={p.src} alt={p.label} style={imgStyle} draggable={false} onLoad={p.onLoad} />
              <span style={{ ...labelStyle, left: 8 }}>{p.label}</span>
            </div>
          ))}
        </div>
      ) : (
        <div
          ref={containerRef}
          style={{ position: "relative", aspectRatio, maxWidth: "100%", background: "#111", touchAction: "none", cursor: "ew-resize" }}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            updateFromPointer(e.clientX);
          }}
          onPointerMove={(e) => {
            if (dragging.current) updateFromPointer(e.clientX);
          }}
          onPointerUp={(e) => {
            dragging.current = false;
            if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
          }}
          onPointerCancel={() => {
            dragging.current = false;
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={beforeSrc} alt="Before" style={imgStyle} draggable={false} onLoad={onBeforeLoad} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={afterSrc} alt="After" style={{ ...imgStyle, clipPath: `inset(0 0 0 ${pos}%)` }} draggable={false} />
          <span style={{ ...labelStyle, left: 8 }}>Before</span>
          <span style={{ ...labelStyle, right: 8 }}>After</span>
          <div
            style={{ position: "absolute", top: 0, bottom: 0, left: `${pos}%`, width: 2, background: "#fff", transform: "translateX(-1px)", pointerEvents: "none" }}
          />
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: `${pos}%`,
              width: 28,
              height: 28,
              borderRadius: "50%",
              background: "#fff",
              border: "2px solid #333",
              transform: "translate(-50%, -50%)",
              pointerEvents: "none",
            }}
          />
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round(pos)}
            aria-label="Before/after divider"
            onChange={(e) => setPos(Number(e.target.value))}
            style={{ position: "absolute", left: 0, right: 0, bottom: 4, width: "100%", margin: 0, opacity: 0.6 }}
          />
        </div>
      )}
    </div>
  );
}
