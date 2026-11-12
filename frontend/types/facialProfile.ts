// Mirrors backend/app/schemas/facial_profile.py. Keep in sync when the API changes.
// All coordinates are original-image pixels (after EXIF orientation), origin top-left.
// Region and zone names are canonical; no model-specific class ids appear here.

import type {
  AnatomicalLayer,
  RegionInfo,
  SegmentationWarning,
  Side,
  Stage,
  Zone,
} from "@/types/faceSegmentation";
import type { FaceMeasurements, Landmark } from "@/types/facialLandmarks";

export type ComponentStatus = "ok" | "degraded" | "unavailable";

export interface ProviderInfo {
  id: string;
  kind: "landmarks" | "parsing" | "zones";
  /** 64 lowercase hex chars, or null. */
  weights_sha256: string | null;
  algorithm_version: string;
  license_warning: string | null;
}

export interface ComponentMeta<S extends ComponentStatus = ComponentStatus> {
  /** Integer version of this component's payload shape. */
  component_version: number;
  status: S;
  /** Uncalibrated. */
  confidence: number | null;
  provider: ProviderInfo;
  reason: string | null;
}

export interface UnavailableComponent {
  meta: ComponentMeta<"unavailable">;
  data: null;
}

export interface AvailableComponent<T> {
  meta: ComponentMeta<"ok" | "degraded">;
  data: T;
}

/**
 * data is null if and only if meta.status is "unavailable".
 * Narrow with isComponentAvailable (lib/facialProfile) or `c.data !== null`,
 * not with `meta.status` directly: TypeScript does not narrow through nested fields.
 */
export type Component<T> = UnavailableComponent | AvailableComponent<T>;

export interface LandmarksData {
  thresholds: {
    min_face_detection_confidence: number;
    min_face_presence_confidence: number;
  };
  /** 478 points: 0-467 face mesh, 468-477 irises. */
  landmarks: Landmark[];
  measurements: FaceMeasurements | null;
}

export interface FaceFrameData {
  ipd_px: number;
  roll_deg: number;
  /** Unit vector, subject's right eye -> left eye (image coordinates). */
  u: [number, number];
  /** Unit vector perpendicular to u, down the face (image coordinates). */
  v: [number, number];
  /** Facial midline points in image pixels. */
  midline: [number, number][];
}

export interface ParsingData {
  parse_window: { x: number; y: number; size: number } | null;
  includes_masks: boolean;
  regions: RegionInfo[];
  /** null when masks are excluded. */
  label_map: AnatomicalLayer | null;
}

export interface ZonesData {
  zones: Zone[];
  stages: Stage[];
}

export interface ComponentSet {
  landmarks: Component<LandmarksData>;
  frame: Component<FaceFrameData>;
  parsing: Component<ParsingData>;
  zones: Component<ZonesData>;
}

export interface ProfileQuality {
  alignment: { passed: boolean; fraction: number } | null;
  warnings: SegmentationWarning[];
  /** Soft hint only. Empty unless status is "ok". */
  retake_suggested: "glasses"[];
}

interface FacialProfileBase {
  /** MAJOR.MINOR */
  schema_version: string;
  message: string | null;
  image: { width: number; height: number };
  quality: ProfileQuality;
  timings_ms: Record<string, number>;
}

export interface FacialProfileOk extends FacialProfileBase {
  status: "ok";
  components: ComponentSet;
}

export interface FacialProfileNoResult extends FacialProfileBase {
  status: "no_face_detected" | "multiple_faces_detected";
  components: null;
}

export type FacialProfile = FacialProfileOk | FacialProfileNoResult;

export type { Side };
