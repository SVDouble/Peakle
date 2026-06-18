"""Annotation overlay drawing."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from peakle.domain.annotations import PeakAnnotation


class AnnotationOverlay:
    """Draws peak annotations onto rendered images."""

    def draw(
        self,
        image: Image.Image,
        annotations: list[PeakAnnotation],
    ) -> Image.Image:
        """Draws visible peak labels.

        Args:
            image: Source RGB image.
            annotations: Peak annotations.

        Returns:
            Annotated image copy.
        """

        output = image.copy()
        draw = ImageDraw.Draw(output, "RGBA")
        font = ImageFont.load_default()

        for annotation in annotations:
            if not annotation.visible:
                continue
            anchor = annotation.anchor
            box = annotation.label_box
            line_start = (box.x_px + box.width_px / 2.0, box.y_px + box.height_px)
            line_end = (anchor.x_px, anchor.y_px)
            draw.line([line_start, line_end], fill=(25, 24, 20, 220), width=2)
            draw.ellipse(
                [anchor.x_px - 3, anchor.y_px - 3, anchor.x_px + 3, anchor.y_px + 3],
                fill=(255, 230, 112, 255),
                outline=(25, 24, 20, 240),
                width=1,
            )
            draw.rounded_rectangle(
                [
                    box.x_px,
                    box.y_px,
                    box.x_px + box.width_px,
                    box.y_px + box.height_px,
                ],
                radius=3,
                fill=(20, 22, 20, 210),
                outline=(255, 230, 112, 230),
                width=1,
            )
            draw.text(
                (box.x_px + 8, box.y_px + 5),
                annotation.peak_name,
                font=font,
                fill=(255, 249, 226, 255),
            )
        return output
