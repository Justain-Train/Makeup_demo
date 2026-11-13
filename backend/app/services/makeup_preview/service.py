"""AdaptedMakeupPlan + selfie -> PreviewResponse (lossless PNG, base64). Pure; never logs pixel data."""

import base64

import cv2
import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.schemas.makeup_preview import ComponentRenderReport, PreviewResponse
from app.services.makeup_preview.renderer import render_layers


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_preview(selfie_rgb: np.ndarray, plan: AdaptedMakeupPlan) -> PreviewResponse:
    height, width = selfie_rgb.shape[:2]
    warnings: list[str] = []
    reports: tuple[ComponentRenderReport, ...] = ()
    image = selfie_rgb

    if plan.status != "ok" or not plan.components:
        warnings.append(f"no components rendered: plan status {plan.status}")
    else:
        result = render_layers(selfie_rgb, plan)
        image, reports = result.image, result.reports
        if reports and all(r.status == "skipped" for r in reports):
            warnings.append("every component was skipped; the preview is unchanged")

    return PreviewResponse(
        image_png_base64=_encode_png_base64(image),
        width=width,
        height=height,
        plan_status=plan.status,
        reports=list(reports),
        warnings=warnings,
    )
