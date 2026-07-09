"""PAGE XML export for page geometry and approved transcription."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID
from xml.etree.ElementTree import Element, QName, SubElement, register_namespace, tostring

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Line, TranscriptionKind
from backend.users.infrastructure.orm_models import User

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
PAGE_XML_SCHEMA = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

register_namespace("", PAGE_XML_NAMESPACE)
register_namespace("xsi", XSI_NAMESPACE)


class PageXmlExportService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()

    async def export_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        document = await self._document_service.get_document(
            session, user, project_id, document_id
        )
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")

        return self._export_part_lines(part, await self._documents.list_part_lines(session, part.id))

    async def export_part_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        part = await self._document_service.get_published_part(
            session, project_id, document_id, part_id
        )
        return self._export_part_lines(part, await self._documents.list_part_lines(session, part.id))

    def _export_part_lines(self, part, lines: list[Line]) -> bytes:
        root = Element(
            _page_tag("PcGts"),
            {
                _xsi_tag("schemaLocation"): PAGE_XML_SCHEMA,
                "pcGtsId": f"pcgts_{part.id}",
            },
        )
        metadata = SubElement(root, _page_tag("Metadata"))
        SubElement(metadata, _page_tag("Creator")).text = "Nomicous"
        SubElement(metadata, _page_tag("Created")).text = datetime.now(UTC).isoformat()

        page_attrs = {
            "imageFilename": part.image_key,
        }
        if part.width is not None:
            page_attrs["imageWidth"] = str(part.width)
        if part.height is not None:
            page_attrs["imageHeight"] = str(part.height)
        page = SubElement(root, _page_tag("Page"), page_attrs)
        region = SubElement(
            page,
            _page_tag("TextRegion"),
            {
                "id": f"region_{part.id}",
                "type": "paragraph",
            },
        )

        for line in lines:
            self._append_text_line(region, line)

        return tostring(root, encoding="utf-8", xml_declaration=True)

    def _append_text_line(self, region: Element, line: Line) -> None:
        text_line = SubElement(
            region,
            _page_tag("TextLine"),
            {
                "id": f"line_{line.id}",
                "custom": f"source:{line.source.value}; kind:{line.kind.value}",
            },
        )
        SubElement(text_line, _page_tag("Coords"), {"points": _points(line.points)})

        baseline_points = _baseline_export_points(line)
        if baseline_points:
            SubElement(text_line, _page_tag("Baseline"), {"points": _points(baseline_points)})

        text_equiv = SubElement(text_line, _page_tag("TextEquiv"))
        SubElement(text_equiv, _page_tag("Unicode")).text = _ground_truth_text(line)


def _page_tag(name: str) -> str:
    return str(QName(PAGE_XML_NAMESPACE, name))


def _xsi_tag(name: str) -> str:
    return str(QName(XSI_NAMESPACE, name))


def _points(points: list[list[float]]) -> str:
    return " ".join(f"{_format_coord(point[0])},{_format_coord(point[1])}" for point in points)


def _format_coord(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _baseline_export_points(line: Line) -> list[list[float]]:
    """Prefer mask polygon for PAGE Baseline; fall back to stored baseline or coords."""
    mask_points = _geometry_points(line.mask)
    if mask_points:
        return mask_points
    baseline_points = _geometry_points(line.baseline)
    if baseline_points:
        return baseline_points
    return [[float(point[0]), float(point[1])] for point in line.points]


def _geometry_points(geometry: dict | None) -> list[list[float]]:
    if not geometry:
        return []
    points = geometry.get("points")
    if not isinstance(points, list):
        return []
    return [
        point
        for point in points
        if isinstance(point, list) and len(point) == 2 and _is_number_pair(point)
    ]


def _is_number_pair(point: list[object]) -> bool:
    return all(isinstance(value, (int, float)) for value in point)


def _ground_truth_text(line: Line) -> str:
    for transcription in line.transcriptions:
        if transcription.transcription.kind == TranscriptionKind.ground_truth:
            return transcription.text
    return ""
