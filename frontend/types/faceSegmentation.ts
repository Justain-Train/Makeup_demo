// Mirrors backend/app/schemas/face_segmentation.py. Keep in sync when the API changes.
// All coordinates are original-image pixels (after EXIF orientation), origin top-left.
// Region and zone names are canonical; no model-specific class ids appear here.

import type { FacialLandmarkResponse } from "@/types/facialLandmarks";

export type Side = "left" | "right";
export type ZoneStatus = "ok" | "low_confidence" | "unavailable";

export interface ModelInfo {
  id: string;
  architecture: string;
  weights_sha256: string;
  input_size: number;
  license_warning: string;
}

export interface RegionInfo {
  /** Canonical region id = pixel value in the label map. */
  id: number;
  name: string;
  /** Landmark convention: "right" = side containing landmarks 33/133/468 (image-left unless the photo is mirrored). */
  side: Side | null;
  source: "segmentation";
  pixel_count: number;
  area_fraction: number;
  /** Mean softmax probability of the winning class. Uncalibrated. */
  mean_probability: number | null;
  present: boolean;
}

export interface AnatomicalLayer {
  encoding: "png_label_map_v1";
  origin: { x: number; y: number };
  width: number;
  height: number;
  not_parsed_value: number;
  label_map_png_base64: string;
  regions: RegionInfo[];
}

export interface ZoneBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Zone {
  id: string;
  name: string;
  group: string;
  side: Side | null;
  sources: ("segmentation" | "landmarks")[];
  status: ZoneStatus;
  reason: string | null;
  /** Mean parser probability, or landmark/parser IoU. Uncalibrated. */
  confidence: number | null;
  coverage: number | null;
  clipped_to: string[];
  bbox: ZoneBBox | null;
  pixel_count: number;
  mask: { encoding: "png_bbox_v1"; png_base64: string } | null;
  polygon: [number, number][] | null;
}

export interface Stage {
  number: number;
  key: string;
  name: string;
  /** Product layer key -> zone ids, in display order. An empty list = zones not implemented yet (Phase 2). */
  layers: Record<string, string[]>;
}

export interface SegmentationWarning {
  code: string;
  message: string;
}

export interface FaceSegmentationResponse {
  status: "ok" | "no_face_detected" | "multiple_faces_detected";
  message: string | null;
  image: { width: number; height: number };
  model: ModelInfo;
  landmarks: FacialLandmarkResponse;
  parse_window: { x: number; y: number; size: number } | null;
  anatomical: AnatomicalLayer | null;
  zones: Zone[];
  stages: Stage[];
  alignment_check: { passed: boolean; fraction: number } | null;
  warnings: SegmentationWarning[];
  retake_suggested?: "glasses"[];
  timings_ms: Record<string, number>;
}
