// Mirrors backend/app/schemas/*.py. Keep in sync when the API changes.

export interface Landmark {
  index: number;
  /** Normalized [0,1] by image width, origin top-left. */
  x: number;
  /** Normalized [0,1] by image height, origin top-left. */
  y: number;
  /** Relative depth; smaller = closer to camera. */
  z: number;
}

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  width: number;
  height: number;
}

/** Pixel measurements. left/right = the subject's own left/right. */
export interface FaceMeasurements {
  face_bounding_box: BoundingBox;
  interpupillary_distance_px: number;
  inner_eye_corner_distance_px: number;
  left_eye_width_px: number;
  right_eye_width_px: number;
  mouth_width_px: number;
  face_width_px: number;
  face_height_px: number;
}

export interface FacialLandmarkResponse {
  face_detected: boolean;
  status: "ok" | "no_face_detected" | "multiple_faces_detected";
  message: string | null;
  /** Faces found by MediaPipe (0-2). */
  face_count: number;
  image: { width: number; height: number };
  thresholds: {
    min_face_detection_confidence: number;
    min_face_presence_confidence: number;
  };
  landmark_count: number;
  landmarks: Landmark[];
  measurements: FaceMeasurements | null;
}

export type Connection = [number, number];

export interface FaceMeshTopology {
  tesselation: Connection[];
  contours: Record<string, Connection[]>;
}
