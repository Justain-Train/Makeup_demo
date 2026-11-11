// Decoding and compositing of segmentation masks. Pure canvas work, no React.
// PNGs are decoded with createImageBitmap(Blob); nothing is rendered as HTML.

import type { AnatomicalLayer, FaceSegmentationResponse, RegionInfo, Zone } from "@/types/faceSegmentation";

export interface DecodedZoneMask {
  x: number;
  y: number;
  width: number;
  height: number;
  /** 1 where the zone covers the pixel. Length = width * height. */
  data: Uint8Array;
}

export interface DecodedSegmentation {
  /** Image pixel of labels[0]. */
  originX: number;
  originY: number;
  width: number;
  height: number;
  /** Canonical region id per pixel (255 = not parsed). */
  labels: Uint8Array;
  zoneMasks: Map<string, DecodedZoneMask>;
}

function base64ToBlob(b64: string): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: "image/png" });
}

/** Decode an 8-bit grayscale PNG into its raw byte values (one per pixel). */
async function decodeGrayPng(b64: string): Promise<{ width: number; height: number; values: Uint8Array }> {
  // No colour management: the pixel values are labels, not colours.
  const bitmap = await createImageBitmap(base64ToBlob(b64), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const values = new Uint8Array(bitmap.width * bitmap.height);
    for (let i = 0; i < values.length; i++) values[i] = rgba[i * 4];
    return { width: bitmap.width, height: bitmap.height, values };
  } finally {
    bitmap.close();
  }
}

export async function decodeSegmentation(seg: FaceSegmentationResponse): Promise<DecodedSegmentation | null> {
  const a: AnatomicalLayer | null = seg.anatomical;
  if (!a) return null;
  const map = await decodeGrayPng(a.label_map_png_base64);
  if (map.width !== a.width || map.height !== a.height) {
    throw new Error(`Label map is ${map.width}x${map.height}, expected ${a.width}x${a.height}`);
  }
  // Sanity check that grayscale values survived decoding unchanged.
  const known = new Set<number>([a.not_parsed_value, ...a.regions.map((r) => r.id)]);
  const seen = new Set<number>(map.values);
  for (const v of seen) if (!known.has(v)) throw new Error(`Label map contains unknown value ${v}`);

  const zoneMasks = new Map<string, DecodedZoneMask>();
  for (const z of seg.zones) {
    if (!z.mask || !z.bbox) continue;
    const m = await decodeGrayPng(z.mask.png_base64);
    if (m.width !== z.bbox.width || m.height !== z.bbox.height) {
      throw new Error(`Zone ${z.id} mask does not match its bbox`);
    }
    const data = new Uint8Array(m.values.length);
    for (let i = 0; i < data.length; i++) data[i] = m.values[i] > 127 ? 1 : 0;
    zoneMasks.set(z.id, { x: z.bbox.x, y: z.bbox.y, width: m.width, height: m.height, data });
  }
  return { originX: a.origin.x, originY: a.origin.y, width: a.width, height: a.height, labels: map.values, zoneMasks };
}

// ---- colours ---------------------------------------------------------------
// The subject's left and right use different hues: right = blue family,
// left = orange family. Midline regions use fixed distinct colours.

const LEFT_HUE: [number, number, number] = [255, 122, 0];
const RIGHT_HUE: [number, number, number] = [28, 126, 214];
const MIDLINE_COLORS: Record<string, [number, number, number]> = {
  skin: [255, 212, 59],
  nose: [32, 201, 151],
  lip_upper: [240, 62, 62],
  lip_lower: [214, 51, 108],
  mouth_interior: [250, 176, 5],
  hair: [134, 94, 60],
  neck: [148, 216, 45],
  cloth: [132, 94, 247],
  hat: [34, 139, 34],
  glasses: [18, 184, 134],
  earring: [250, 82, 82],
  necklace: [252, 194, 215],
  background: [134, 142, 150],
  face_skin: [255, 212, 59],
  forehead: [255, 169, 77],
  chin: [190, 75, 219],
  jawline: [240, 62, 62],
  lip_full: [240, 62, 62],
  lip_outer_boundary: [255, 255, 255],
  lip_inner_boundary: [0, 0, 0],
};

export type RGB = [number, number, number];

/** Left/right-aware colour for a canonical region or zone name/side. */
export function colorFor(name: string, side: "left" | "right" | null): RGB {
  if (side === "left") return shade(LEFT_HUE, name);
  if (side === "right") return shade(RIGHT_HUE, name);
  return MIDLINE_COLORS[name] ?? [200, 200, 200];
}

// Different features on the same side get slightly different shades.
function shade(base: RGB, name: string): RGB {
  let h = 0;
  for (const ch of name.replace(/_(l|r)$/, "")) h = (h * 31 + ch.charCodeAt(0)) % 7;
  const f = 0.65 + h * 0.07;
  return [Math.min(255, base[0] * f), Math.min(255, base[1] * f), Math.min(255, base[2] * f)].map(Math.round) as RGB;
}

export function cssColor(c: RGB): string {
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// ---- compositing ---------------------------------------------------------------

export interface OverlaySelection {
  regions: ReadonlySet<number>;
  zones: ReadonlySet<string>;
}

export interface Overlay {
  canvas: HTMLCanvasElement;
  /** Image pixel of the canvas' top-left. */
  x: number;
  y: number;
}

/**
 * Compose the selected regions (filled) and zones (filled + outline) into one
 * canvas covering the parsed area. Opacity is applied when drawing, so only
 * selection changes require calling this again.
 */
export function composeOverlay(
  seg: FaceSegmentationResponse,
  decoded: DecodedSegmentation,
  selection: OverlaySelection,
): Overlay | null {
  const regionInfo = new Map<number, RegionInfo>(seg.anatomical?.regions.map((r) => [r.id, r]) ?? []);
  if (selection.regions.size === 0 && selection.zones.size === 0) return null;
  const { width, height, labels } = decoded;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(width, height);
  const px = img.data;

  // Regions: lookup table id -> rgb, then one pass over the label map.
  const lut = new Uint8Array(256 * 3);
  const on = new Uint8Array(256);
  for (const id of selection.regions) {
    const info = regionInfo.get(id);
    if (!info) continue;
    const c = colorFor(info.name, info.side);
    lut.set(c, id * 3);
    on[id] = 1;
  }
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i];
    if (on[id]) {
      px[i * 4] = lut[id * 3];
      px[i * 4 + 1] = lut[id * 3 + 1];
      px[i * 4 + 2] = lut[id * 3 + 2];
      px[i * 4 + 3] = 255;
    }
  }

  // Zones: fill plus a 1-2 px outline (thicker on bigger images).
  const outline = Math.max(width, height) > 1500 ? 2 : 1;
  const zoneById = new Map<string, Zone>(seg.zones.map((z) => [z.id, z]));
  for (const id of selection.zones) {
    const z = zoneById.get(id);
    const m = decoded.zoneMasks.get(id);
    if (!z || !m) continue;
    const c = colorFor(z.id.replace(/_(l|r)$/, ""), z.side);
    const x0 = m.x - decoded.originX;
    const y0 = m.y - decoded.originY;
    for (let y = 0; y < m.height; y++) {
      for (let x = 0; x < m.width; x++) {
        if (!m.data[y * m.width + x]) continue;
        const ox = x0 + x;
        const oy = y0 + y;
        if (ox < 0 || oy < 0 || ox >= width || oy >= height) continue;
        const edge = isEdge(m, x, y, outline);
        const o = (oy * width + ox) * 4;
        // Edge pixels are drawn in the full colour; inside pixels blend with what is there.
        if (edge || px[o + 3] === 0) {
          px[o] = c[0];
          px[o + 1] = c[1];
          px[o + 2] = c[2];
          px[o + 3] = edge ? 255 : 170;
        } else {
          px[o] = (px[o] + c[0]) >> 1;
          px[o + 1] = (px[o + 1] + c[1]) >> 1;
          px[o + 2] = (px[o + 2] + c[2]) >> 1;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return { canvas, x: decoded.originX, y: decoded.originY };
}

function isEdge(m: DecodedZoneMask, x: number, y: number, r: number): boolean {
  for (let d = 1; d <= r; d++) {
    if (x - d < 0 || y - d < 0 || x + d >= m.width || y + d >= m.height) return true;
    if (
      !m.data[y * m.width + x - d] ||
      !m.data[y * m.width + x + d] ||
      !m.data[(y - d) * m.width + x] ||
      !m.data[(y + d) * m.width + x]
    ) {
      return true;
    }
  }
  return false;
}

/** Region name and the zones covering an image pixel, for the hover tooltip. */
export function describePixel(
  seg: FaceSegmentationResponse,
  decoded: DecodedSegmentation,
  imgX: number,
  imgY: number,
): string | null {
  const x = Math.floor(imgX) - decoded.originX;
  const y = Math.floor(imgY) - decoded.originY;
  if (x < 0 || y < 0 || x >= decoded.width || y >= decoded.height) return null;
  const id = decoded.labels[y * decoded.width + x];
  const region = seg.anatomical?.regions.find((r) => r.id === id);
  const zones: string[] = [];
  for (const [zid, m] of decoded.zoneMasks) {
    const zx = Math.floor(imgX) - m.x;
    const zy = Math.floor(imgY) - m.y;
    if (zx >= 0 && zy >= 0 && zx < m.width && zy < m.height && m.data[zy * m.width + zx]) zones.push(zid);
  }
  return `region: ${region?.name ?? "not parsed"}\nzones: ${zones.length ? zones.join(", ") : "-"}`;
}
