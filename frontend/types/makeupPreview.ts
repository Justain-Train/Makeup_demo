export type SkipReason = "no_mask" | "mask_invalid" | "no_colour" | "zero_opacity" | "empty_mask";

export interface ComponentRenderReport {
  index: number;
  status: "rendered" | "skipped";
  reason: SkipReason | null;
  opacity: number;
  finish_applied: string;
  directional_stretch: "deferred" | "applied" | "not_applicable";
  pixels_touched: number;
}

export interface PreviewResponse {
  schema_version: "1.0";
  image_png_base64: string;
  width: number;
  height: number;
  plan_status: string;
  reports: ComponentRenderReport[];
  warnings: string[];
}
