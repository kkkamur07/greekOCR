import { useEffect, useState, type MouseEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Button, Card, Space, Spin, Typography } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type LinePoint,
  type LineResponse,
  type LineUpsertRequest,
  type LinesReplaceRequest,
  type LayoutPoint,
  type PartLayoutResponse,
} from '../api/client';
import { ApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';
import { AuthenticatedImage } from '../components/AuthenticatedImage';

function accessMessage(error: ApiError): string {
  if (error.status === 403 || error.status === 404) {
    return 'This page is not available to your account.';
  }
  return error.message;
}

function layoutMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return 'Only project members can edit layout.';
  }
  return error instanceof Error ? error.message : 'Layout update failed.';
}

export function PageEditorPlaceholderPage() {
  const { projectId, documentId, partId } = useParams<{
    projectId: string;
    documentId: string;
    partId: string;
  }>();
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [part, setPart] = useState<DocumentPartResponse | null>(null);
  const [layout, setLayout] = useState<PartLayoutResponse>({ blocks: [], lines: [] });
  const [lines, setLines] = useState<LineResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [lineError, setLineError] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState<'none' | 'rectangle' | 'polygon'>('none');
  const [draftStart, setDraftStart] = useState<LayoutPoint | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<LinePoint[]>([]);
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [selectedLineSnapshot, setSelectedLineSnapshot] = useState<{
    baseline?: LayoutPoint[] | null;
    mask?: LayoutPoint[] | null;
  } | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId || !documentId || !partId) {
      setLoading(false);
      setError('Page route is incomplete.');
      return;
    }

    setLoading(true);
    setError(null);
    setLayoutError(null);
    setLineError(null);
    setDocument(null);
    setPart(null);
    setLayout({ blocks: [], lines: [] });
    setLines([]);

    (async () => {
      try {
        const doc = await api.getDocument(projectId, documentId);
        const sortedParts = [...doc.parts].sort((a, b) => a.order - b.order);
        const selectedPart = sortedParts.find((item) => item.id === partId);
        if (!selectedPart) {
          setError('This document part was not found.');
          return;
        }
        setDocument(doc);
        setPart(selectedPart);
        try {
          const loadedLayout = await api.getPartLayout(projectId, documentId, partId);
          setLayout(loadedLayout ?? { blocks: [], lines: [] });
        } catch (err) {
          setLayoutError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Layout editing is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load layout.',
          );
        }
        try {
          setLines(await api.listPartLines(projectId, documentId, partId));
        } catch (err) {
          setLineError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Segment geometry is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load Segment geometry.',
          );
        }
      } catch (err) {
        setError(err instanceof ApiError ? accessMessage(err) : 'Failed to load page.');
      } finally {
        setLoading(false);
      }
    })();
  }, [projectId, documentId, partId]);

  const partIndex =
    document && part
      ? [...document.parts].sort((a, b) => a.order - b.order).findIndex((item) => item.id === part.id) +
        1
      : null;

  function moveSelectedBaseline(deltaY: number) {
    if (!selectedLineId) return;
    setSaveMessage(null);
    setMutationError(null);
    setLayout((current) => ({
      ...current,
      lines: current.lines.map((line) =>
        line.id === selectedLineId
          ? {
              ...line,
              baseline: (line.baseline ?? []).map(([x, y]) => [x, y + deltaY]),
            }
          : line,
      ),
    }));
  }

  async function saveSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const selectedLine = layout.lines.find((line) => line.id === selectedLineId);
    if (!selectedLine) return;

    try {
      await api.updateLineGeometry(projectId, documentId, partId, selectedLineId, {
        baseline: selectedLine.baseline,
        mask: selectedLine.mask,
      });
      setLayout((current) => ({
        ...current,
        lines: current.lines.map((line) =>
          line.id === selectedLineId ? { ...line, manual_geometry: true } : line,
        ),
      }));
      setMutationError(null);
      setSaveMessage('Manual geometry saved');
      setSelectedLineSnapshot({
        baseline: selectedLine.baseline,
        mask: selectedLine.mask,
      });
    } catch (err) {
      if (selectedLineSnapshot) {
        setLayout((current) => ({
          ...current,
          lines: current.lines.map((line) =>
            line.id === selectedLineId
              ? {
                  ...line,
                  baseline: selectedLineSnapshot.baseline,
                  mask: selectedLineSnapshot.mask,
                }
              : line,
          ),
        }));
      }
      setSaveMessage(null);
      setMutationError(layoutMutationMessage(err));
    }
  }

  async function resetSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const resetLayout = await api.resetPartLayout(projectId, documentId, partId, {
      line_ids: [selectedLineId],
    });
    setLayout(resetLayout ?? { blocks: [], lines: [] });
    setSelectedLineSnapshot(null);
    setSaveMessage('Layout reset');
  }

  async function replaceWithManualLine(kind: 'rectangle' | 'polygon', points: LinePoint[]) {
    if (!projectId || !documentId || !partId) return;
    const existing = [...lines]
      .sort((a, b) => a.order - b.order)
      .map<LineUpsertRequest>((line, order) => ({
        id: line.id,
        order,
        kind: line.kind,
        points: line.points,
        source: line.source,
      }));
    const newLine: LineUpsertRequest = {
      order: existing.length,
      kind,
      points,
      source: 'manual',
    };
    const body: LinesReplaceRequest = {
      lines: [...existing, newLine],
    };
    const saved = await api.replacePartLines(projectId, documentId, partId, body);
    setLines(saved);
    setDrawMode('none');
  }

  async function moveSelectedSegmentRight() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const updatedLines = [...lines]
      .sort((a, b) => a.order - b.order)
      .map<LineUpsertRequest>((line, order) => ({
        id: line.id,
        order,
        kind: line.kind,
        points:
          line.id === selectedSegmentId
            ? line.points.map(([x, y]) => [x + 5, y])
            : line.points,
        source: line.source,
      }));
    const saved = await api.replacePartLines(projectId, documentId, partId, {
      lines: updatedLines,
    });
    setLines(saved ?? lines.map((line) =>
      line.id === selectedSegmentId
        ? { ...line, points: line.points.map(([x, y]) => [x + 5, y]) }
        : line,
    ));
  }

  async function deleteSelectedSegment() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const remainingLines = [...lines]
      .sort((a, b) => a.order - b.order)
      .filter((line) => line.id !== selectedSegmentId)
      .map<LineUpsertRequest>((line, order) => ({
        id: line.id,
        order,
        kind: line.kind,
        points: line.points,
        source: line.source,
      }));
    const saved = await api.replacePartLines(projectId, documentId, partId, {
      lines: remainingLines,
    });
    setLines(saved ?? lines.filter((line) => line.id !== selectedSegmentId));
    setSelectedSegmentId(null);
  }

  function pickDrawMode(nextMode: 'rectangle' | 'polygon') {
    setDrawMode((mode) => (mode === nextMode ? 'none' : nextMode));
    setDraftPolygon([]);
    setDraftStart(null);
  }

  return (
    <AppLayout
      title={partIndex ? `Page ${partIndex}` : 'Page editor'}
      extra={
        projectId && documentId ? (
          <Link to={`/projects/${projectId}/documents/${documentId}`}>
            <Button icon={<ArrowLeftOutlined />}>Document parts</Button>
          </Link>
        ) : null
      }
    >
      {loading && (
        <Space>
          <Spin />
          <Typography.Text>Loading page...</Typography.Text>
        </Space>
      )}

      {!loading && error && (
        <Alert
          type="warning"
          showIcon
          message="Page unavailable"
          description={error}
        />
      )}

      {!loading && document && part && (
        <Card
          style={{
            background: '#15181d',
            borderColor: '#2f3540',
            color: '#f7f2e8',
          }}
          styles={{
            body: { display: 'grid', gap: 16 },
          }}
        >
          <Space direction="vertical" size={4}>
            <Typography.Text style={{ color: '#d8c7a1', letterSpacing: 1 }}>
              ANNOTE PAGE WORKSPACE
            </Typography.Text>
            <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
              {document.name} · Page {partIndex}
            </Typography.Title>
            <Typography.Title level={5} style={{ color: '#f7f2e8', margin: 0 }}>
              Layout edit
            </Typography.Title>
            <Typography.Text style={{ color: '#c5ccd6' }}>
              Correct Segment baselines and block geometry without entering transcription mode.
            </Typography.Text>
          </Space>

          <Space wrap>
            <Button
              type={drawMode === 'rectangle' ? 'primary' : 'default'}
              onClick={() => pickDrawMode('rectangle')}
            >
              Rectangle segment
            </Button>
            <Button
              type={drawMode === 'polygon' ? 'primary' : 'default'}
              onClick={() => pickDrawMode('polygon')}
            >
              Polygon segment
            </Button>
            <Typography.Text style={{ color: '#c5ccd6' }}>
              {lines.length} {lines.length === 1 ? 'Segment' : 'Segments'}
            </Typography.Text>
            {selectedSegmentId && (
              <>
                <Button onClick={() => void moveSelectedSegmentRight()}>
                  Move segment right
                </Button>
                <Button danger onClick={() => void deleteSelectedSegment()}>
                  Delete Segment
                </Button>
              </>
            )}
            {selectedLineId && (
              <>
                <Button onClick={() => moveSelectedBaseline(5)}>Move baseline down</Button>
                <Button type="primary" onClick={() => void saveSelectedLine()}>
                  Save layout
                </Button>
                <Button danger onClick={() => void resetSelectedLine()}>
                  Reset layout
                </Button>
              </>
            )}
          </Space>

          {saveMessage && <Alert type="success" showIcon message={saveMessage} />}
          {mutationError && <Alert type="error" showIcon message={mutationError} />}

          {layoutError && (
            <Alert
              type="warning"
              showIcon
              message="Layout API unavailable"
              description={layoutError}
            />
          )}
          {lineError && (
            <Alert
              type="warning"
              showIcon
              message="Segment API unavailable"
              description={lineError}
            />
          )}

          <div
            style={{
              border: '1px solid #3b4350',
              borderRadius: 8,
              padding: 16,
              background: '#0f1115',
              overflow: 'auto',
            }}
          >
            <LayoutCanvas
              imageUrl={part.image_url}
              imageAlt={`Page ${partIndex}`}
              imageWidth={part.width ?? 640}
              imageHeight={part.height ?? 900}
              layout={layout}
              lines={lines}
              drawingRectangle={drawMode === 'rectangle'}
              drawingPolygon={drawMode === 'polygon'}
              onDraftStart={setDraftStart}
              onRectangleDrawn={async (end) => {
                if (!draftStart) return;
                const rectangle = rectanglePoints(draftStart, end);
                await replaceWithManualLine('rectangle', rectangle);
                setDraftStart(null);
              }}
              onPolygonPoint={(point) => setDraftPolygon((current) => [...current, point])}
              onPolygonComplete={async () => {
                if (draftPolygon.length < 3) return;
                await replaceWithManualLine('polygon', draftPolygon);
                setDraftPolygon([]);
              }}
              onSelectLine={(lineId) => {
                const selectedLine = layout.lines.find((line) => line.id === lineId);
                setSelectedLineId(lineId);
                setSelectedSegmentId(null);
                setSelectedLineSnapshot(
                  selectedLine
                    ? { baseline: selectedLine.baseline, mask: selectedLine.mask }
                    : null,
                );
                setSaveMessage(null);
              }}
              onSelectSegment={(lineId) => {
                setSelectedSegmentId(lineId);
                setSelectedLineId(null);
                setSaveMessage(null);
              }}
            />
          </div>
        </Card>
      )}
    </AppLayout>
  );
}

function points(points: LayoutPoint[] | null | undefined): string {
  return (points ?? []).map(([x, y]) => `${x},${y}`).join(' ');
}

function rectanglePoints(start: LinePoint, end: LinePoint): LinePoint[] {
  const [startX, startY] = start;
  const [endX, endY] = end;
  return [
    [Math.min(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.max(startY, endY)],
    [Math.min(startX, endX), Math.max(startY, endY)],
  ];
}

function LayoutCanvas({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  drawingRectangle,
  drawingPolygon,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
}: {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  onDraftStart: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
}) {
  const eventPoint = (event: MouseEvent<SVGSVGElement>): LinePoint => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return [event.clientX, event.clientY];
    }
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * imageWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * imageHeight),
    ];
  };

  return (
    <div style={{ position: 'relative', width: 480, maxWidth: '100%' }}>
      <AuthenticatedImage src={imageUrl} alt={imageAlt} width={480} />
      <svg
        role="img"
        aria-label="Page geometry canvas"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        onMouseDown={(event) => {
          if (drawingRectangle) onDraftStart(eventPoint(event));
        }}
        onMouseUp={(event) => {
          if (drawingRectangle) onRectangleDrawn(eventPoint(event));
        }}
        onClick={(event) => {
          if (drawingPolygon) onPolygonPoint(eventPoint(event));
        }}
        onDoubleClick={() => {
          if (drawingPolygon) onPolygonComplete();
        }}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          cursor: drawingRectangle || drawingPolygon ? 'crosshair' : 'default',
        }}
      >
        {layout.blocks.map((block) => (
          <polygon
            key={block.id}
            aria-label={`Block ${block.id}`}
            points={points(block.box)}
            fill="rgba(216, 199, 161, 0.08)"
            stroke="#d8c7a1"
            strokeWidth={4}
          />
        ))}
        {layout.lines.map((line) => (
          <polyline
            key={line.id}
            aria-label={`Line ${line.id} baseline`}
            onClick={() => onSelectLine(line.id)}
            points={points(line.baseline)}
            fill="none"
            stroke={line.manual_geometry ? '#58d68d' : '#5dade2'}
            strokeLinecap="round"
            strokeWidth={6}
          />
        ))}
        {lines
          .slice()
          .sort((a, b) => a.order - b.order)
          .map((line) => (
            <polygon
              key={line.id}
              aria-label={`Segment ${line.order + 1}`}
              onClick={() => onSelectSegment(line.id)}
              points={points(line.points)}
              fill="rgba(88, 214, 141, 0.14)"
              stroke="#58d68d"
              strokeWidth={5}
            />
          ))}
      </svg>
    </div>
  );
}
