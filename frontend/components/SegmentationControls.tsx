"use client";

import { colorFor, cssColor } from "@/lib/segmentation";
import type { FaceSegmentationResponse, RegionInfo, Zone } from "@/types/faceSegmentation";

export interface SegmentationSelection {
  regions: ReadonlySet<number>;
  zones: ReadonlySet<string>;
}

interface Props {
  seg: FaceSegmentationResponse;
  selection: SegmentationSelection;
  onSelectionChange: (next: SegmentationSelection) => void;
  opacity: number;
  onOpacityChange: (v: number) => void;
  compareEnabled: boolean;
  onCompareEnabledChange: (v: boolean) => void;
  compare: number;
  onCompareChange: (v: number) => void;
}

const STATUS_COLOR: Record<Zone["status"], string> = {
  ok: "#2b8a3e",
  low_confidence: "#e67700",
  unavailable: "#c92a2a",
};

function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      style={{ display: "inline-block", width: 10, height: 10, background: color, border: "1px solid #0004", marginRight: 4 }}
    />
  );
}

export default function SegmentationControls({
  seg,
  selection,
  onSelectionChange,
  opacity,
  onOpacityChange,
  compareEnabled,
  onCompareEnabledChange,
  compare,
  onCompareChange,
}: Props) {
  const regions: RegionInfo[] = (seg.anatomical?.regions ?? []).filter((r) => r.present);
  const zoneById = new Map(seg.zones.map((z) => [z.id, z]));
  const usable = (ids: string[]) => ids.filter((id) => zoneById.get(id)?.mask);
  const addZones = (ids: string[]) =>
    onSelectionChange({ ...selection, zones: new Set([...selection.zones, ...usable(ids)]) });

  return (
    <div style={{ fontSize: 13, minWidth: 300, maxWidth: 380 }}>
      <p style={{ background: "#fff3bf", padding: 6, borderRadius: 4, margin: "0 0 8px" }} role="note">
        <b>License:</b> {seg.model.license_warning}
      </p>
      <p style={{ margin: "4px 0", color: "#555" }}>
        {seg.model.architecture} · sha256 {seg.model.weights_sha256.slice(0, 10)}… · total{" "}
        {Math.round(seg.timings_ms.total ?? 0)} ms
        {seg.alignment_check && (
          <>
            {" "}
            · alignment{" "}
            <b style={{ color: seg.alignment_check.passed ? "#2b8a3e" : "#c92a2a" }}>
              {seg.alignment_check.passed ? "ok" : "FAILED"}
            </b>{" "}
            ({Math.round(seg.alignment_check.fraction * 100)}%)
          </>
        )}
      </p>

      {seg.warnings.length > 0 && (
        <ul style={{ color: "#e67700", paddingLeft: 18, margin: "6px 0" }}>
          {seg.warnings.map((w, i) => (
            <li key={`${w.code}-${i}`}>
              <code>{w.code}</code>: {w.message}
            </li>
          ))}
        </ul>
      )}

      {seg.retake_suggested?.includes("glasses") && (
        <p role="note" style={{ color: "#e67700", margin: "6px 0" }}>
          To get eye-makeup results, retake the photo without your glasses. You can keep them on, but eye-area results
          are limited while glasses are on.
        </p>
      )}

      <label style={{ display: "block", margin: "8px 0 4px" }}>
        Overlay opacity {Math.round(opacity * 100)}%
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => onOpacityChange(Number(e.target.value))}
          style={{ display: "block", width: "100%" }}
        />
      </label>
      <label style={{ display: "block", margin: "4px 0" }}>
        <input type="checkbox" checked={compareEnabled} onChange={(e) => onCompareEnabledChange(e.target.checked)} /> Compare
        (wipe: overlay on the left of the divider)
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={compare}
          disabled={!compareEnabled}
          aria-label="Compare divider position"
          onChange={(e) => onCompareChange(Number(e.target.value))}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <fieldset style={{ margin: "8px 0", padding: 6 }}>
        <legend>Anatomical regions (parser)</legend>
        <button
          onClick={() => onSelectionChange({ ...selection, regions: new Set(regions.map((r) => r.id)) })}
          style={{ marginRight: 6 }}
        >
          All
        </button>
        <button onClick={() => onSelectionChange({ ...selection, regions: new Set() })}>None</button>
        {regions.map((r) => (
          <label key={r.id} style={{ display: "block" }}>
            <input
              type="checkbox"
              checked={selection.regions.has(r.id)}
              onChange={() => onSelectionChange({ ...selection, regions: toggled(selection.regions, r.id) })}
            />{" "}
            <Swatch color={cssColor(colorFor(r.name, r.side))} />
            {r.name}
            {r.side && <span style={{ color: "#777" }}> ({r.side})</span>}
            <span style={{ color: "#777" }}>
              {" "}
              {(r.area_fraction * 100).toFixed(1)}%
              {r.mean_probability !== null && ` · p ${r.mean_probability.toFixed(2)}`}
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset style={{ margin: "8px 0", padding: 6 }}>
        <legend>Application zones (stage / layer)</legend>
        <button onClick={() => onSelectionChange({ ...selection, zones: new Set() })} style={{ marginBottom: 4 }}>
          Clear all zones
        </button>
        {seg.stages.map((stage) => {
          const stageIds = [...new Set(Object.values(stage.layers).flat())];
          return (
            <details key={stage.key} style={{ marginTop: 4 }}>
              <summary>
                <b>
                  {stage.number}. {stage.name}
                </b>
              </summary>
              <button onClick={() => addZones(stageIds)} disabled={usable(stageIds).length === 0}>
                Select all in stage
              </button>
              {Object.entries(stage.layers).map(([layer, ids]) => (
                <div key={layer} style={{ marginLeft: 10, marginTop: 4 }}>
                  <i>{layer.replace(/_/g, " ")}</i>{" "}
                  {ids.length > 0 && (
                    <button onClick={() => addZones(ids)} disabled={usable(ids).length === 0}>
                      select layer
                    </button>
                  )}
                  {ids.length === 0 && <div style={{ color: "#777" }}>no zones yet (Phase 2)</div>}
                  {ids.map((id) => {
                    const z = zoneById.get(id);
                    if (!z) return null;
                    return (
                      <label key={id} style={{ display: "block" }} title={z.reason ?? undefined}>
                        <input
                          type="checkbox"
                          checked={selection.zones.has(z.id)}
                          disabled={!z.mask}
                          onChange={() => onSelectionChange({ ...selection, zones: toggled(selection.zones, z.id) })}
                        />{" "}
                        <Swatch color={cssColor(colorFor(z.id.replace(/_(l|r)$/, ""), z.side))} />
                        {z.name}{" "}
                        <span style={{ color: STATUS_COLOR[z.status] }}>
                          [{z.status === "low_confidence" ? "low confidence" : z.status}]
                        </span>
                        {z.confidence !== null && <span style={{ color: "#777" }}> {z.confidence.toFixed(2)}</span>}
                        {z.reason && <div style={{ color: "#777", marginLeft: 22 }}>{z.reason}</div>}
                      </label>
                    );
                  })}
                </div>
              ))}
            </details>
          );
        })}
        <p style={{ color: "#777", margin: "6px 0 0" }}>
          Confidence values are uncalibrated (mean parser probability, or landmark/parser IoU).
        </p>
      </fieldset>
    </div>
  );
}
