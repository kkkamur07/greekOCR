import { useEffect, useState, type ChangeEvent, type MouseEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Button, Card, Input, Space, Spin, Typography } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type LineTranscriptionResponse,
  type LinePoint,
  type LineResponse,
  type LineUpsertRequest,
  type LinesReplaceRequest,
  type LayoutPoint,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
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

function reviewMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return 'Only project members can change Review status.';
  }
  return error instanceof Error ? error.message : 'Review status update failed.';
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
  const [editorMode, setEditorMode] = useState<'layout' | 'transcription'>('layout');
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
  const [transcriptionLayers, setTranscriptionLayers] = useState<TranscriptionLayerResponse[]>([]);
  const [selectedTranscriptionLayerId, setSelectedTranscriptionLayerId] = useState<string | null>(null);
  const [transcriptionSaveMessage, setTranscriptionSaveMessage] = useState<string | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const [groundTruthTranscriptionId, setGroundTruthTranscriptionId] = useState<string | null>(null);
  const [pageTranscriptionText, setPageTranscriptionText] = useState('');
  const [approvedTextDraft, setApprovedTextDraft] = useState('');
  const [textLines, setTextLines] = useState<
    { order: number; text: string; paired_line_id: string | null }[]
  >([]);
  const [pairingProgress, setPairingProgress] = useState({
    paired_lines: 0,
    total_lines: 0,
    percent: 0,
  });
  const [pairingError, setPairingError] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);

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
    setEditorMode('layout');
    setTranscriptionLayers([]);
    setSelectedTranscriptionLayerId(null);
    setTranscriptionSaveMessage(null);
    setCopyMessage(null);
    setGroundTruthTranscriptionId(null);
    setTextLines([]);
    setApprovedTextDraft('');
    setPairingProgress({ paired_lines: 0, total_lines: 0, percent: 0 });
    setPairingError(null);
    setReviewError(null);

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
        try {
          const layers = await api.listTranscriptions(projectId, documentId);
          const groundTruth = layers.find((layer) => layer.kind === 'ground_truth');
          setTranscriptionLayers(layers);
          setGroundTruthTranscriptionId(groundTruth?.id ?? null);
          setSelectedTranscriptionLayerId(groundTruth?.id ?? layers[0]?.id ?? null);
          const pairing = await api.getPagePairing(projectId, documentId, partId);
          setTextLines(pairing.text_lines);
          setPairingProgress(pairing.pairing_progress);
        } catch (err) {
          setPairingError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Pairing is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load Pairing progress.',
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

  const selectedSegmentIndex =
    selectedSegmentId === null
      ? null
      : [...lines]
          .sort((a, b) => a.order - b.order)
          .findIndex((line) => line.id === selectedSegmentId);

  const selectedSegmentNumber = selectedSegmentIndex === null || selectedSegmentIndex < 0
    ? null
    : selectedSegmentIndex + 1;

  function approvedText(line: LineResponse): string | null {
    return (
      line.line_transcriptions.find(
        (transcription) => transcription.transcription_kind === 'ground_truth',
      )?.text ?? null
    );
  }

  function lineTextForLayer(line: LineResponse, transcriptionLayerId: string | null): string {
    if (!transcriptionLayerId) return '';
    return (
      line.line_transcriptions.find(
        (transcription) => transcription.transcription_id === transcriptionLayerId,
      )?.text ?? ''
    );
  }

  const selectedTranscriptionLayer =
    selectedTranscriptionLayerId === null
      ? null
      : transcriptionLayers.find((layer) => layer.id === selectedTranscriptionLayerId) ?? null;

  const selectedSegment =
    selectedSegmentId === null ? null : lines.find((line) => line.id === selectedSegmentId) ?? null;

  function upsertLineRequest(line: LineResponse, order: number): LineUpsertRequest {
    const text = approvedText(line);
    const request: LineUpsertRequest = {
      id: line.id,
      order,
      kind: line.kind,
      points: line.points,
      source: line.source,
    };
    if (text !== null) {
      request.approved_text = text;
    }
    return request;
  }

  function withLocalGroundTruth(lineId: string, text: string): LineResponse[] {
    if (!groundTruthTranscriptionId) return lines;
    return lines.map((line) => {
      if (line.id !== lineId) return line;
      const existing = line.line_transcriptions.filter(
        (transcription) => transcription.transcription_kind !== 'ground_truth',
      );
      const nextTranscription: LineTranscriptionResponse = {
        id: line.line_transcriptions.find(
          (transcription) => transcription.transcription_kind === 'ground_truth',
        )?.id ?? `ground-truth-${lineId}`,
        transcription_id: groundTruthTranscriptionId,
        transcription_kind: 'ground_truth',
        text,
        confidence: null,
        text_source: 'human_edited',
        character_confidences: null,
      };
      return { ...line, line_transcriptions: [...existing, nextTranscription] };
    });
  }

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
      .map<LineUpsertRequest>(upsertLineRequest);
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
        ...upsertLineRequest(line, order),
        points:
          line.id === selectedSegmentId
            ? line.points.map(([x, y]) => [x + 5, y])
            : line.points,
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
      .map<LineUpsertRequest>(upsertLineRequest);
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

  async function importPageTranscription() {
    if (!projectId || !documentId || !partId) return;
    try {
      const pairing = await api.importPageTranscription(projectId, documentId, partId, {
        text: pageTranscriptionText,
      });
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to import Page transcription.');
    }
  }

  async function pairTextLine(order: number) {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    try {
      const pairing = await api.pairTextLine(projectId, documentId, partId, {
        line_id: selectedSegmentId,
        text_line_order: order,
      });
      const candidate = pairing.text_lines.find((textLine) => textLine.order === order);
      if (candidate) {
        setLines(withLocalGroundTruth(selectedSegmentId, candidate.text));
        setApprovedTextDraft(candidate.text);
      }
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to pair Text line.');
    }
  }

  async function saveApprovedText() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (!groundTruthTranscriptionId) {
      setPairingError('Ground truth transcription layer is not available.');
      return;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: approvedTextDraft },
      );
      setLines(withLocalGroundTruth(selectedSegmentId, updated.text));
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to save approved text.');
    }
  }

  function selectTranscriptionLayer(event: ChangeEvent<HTMLSelectElement>) {
    const nextLayerId = event.target.value;
    setSelectedTranscriptionLayerId(nextLayerId);
    setTranscriptionSaveMessage(null);
    setCopyMessage(null);
    setPairingError(null);
    if (selectedSegment) {
      setApprovedTextDraft(lineTextForLayer(selectedSegment, nextLayerId));
    }
  }

  async function saveGroundTruthText() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (!groundTruthTranscriptionId || selectedTranscriptionLayer?.kind !== 'ground_truth') {
      setPairingError('Only Ground truth can be edited.');
      return;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: approvedTextDraft },
      );
      setLines(withLocalGroundTruth(selectedSegmentId, updated.text));
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
      setTranscriptionSaveMessage('Ground truth text saved');
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(err instanceof Error ? err.message : 'Failed to save Ground truth text.');
    }
  }

  async function copySelectedLayerToGroundTruth(lineIds: string[] | null) {
    if (!projectId || !documentId || !partId || !selectedTranscriptionLayerId) return;
    if (selectedTranscriptionLayer?.kind !== 'model') {
      setPairingError('Choose a model layer before copying to Ground truth.');
      return;
    }
    try {
      const result = await api.copyToGroundTruth(projectId, documentId, selectedTranscriptionLayerId, {
        line_ids: lineIds,
      });
      const [reloadedLines, pairing] = await Promise.all([
        api.listPartLines(projectId, documentId, partId),
        api.getPagePairing(projectId, documentId, partId),
      ]);
      setLines(reloadedLines);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
      setTranscriptionSaveMessage(null);
      const copiedCount = result.copied_line_ids.length;
      setCopyMessage(
        `Copied ${copiedCount} ${copiedCount === 1 ? 'Segment' : 'Segments'} to Ground truth`,
      );
    } catch (err) {
      setCopyMessage(null);
      setPairingError(err instanceof Error ? err.message : 'Failed to copy to Ground truth.');
    }
  }

  async function updateReviewStatus(reviewed: boolean) {
    if (!projectId || !documentId || !partId) return;
    try {
      const updated = await api.updatePartReviewStatus(projectId, documentId, partId, {
        reviewed,
      });
      setPart(updated);
      setDocument((current) =>
        current
          ? {
              ...current,
              parts: current.parts.map((item) => (item.id === updated.id ? updated : item)),
            }
          : current,
      );
      setReviewError(null);
    } catch (err) {
      setReviewError(reviewMutationMessage(err));
    }
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
              {editorMode === 'layout' ? 'Layout edit' : 'Transcription edit'}
            </Typography.Title>
            <Typography.Text style={{ color: '#c5ccd6' }}>
              {editorMode === 'layout'
                ? 'Correct Segment baselines and block geometry without entering transcription mode.'
                : 'Edit the Ground truth transcription while model layers stay read-only.'}
            </Typography.Text>
          </Space>

          <Space wrap>
            <Button
              type={editorMode === 'layout' ? 'primary' : 'default'}
              onClick={() => setEditorMode('layout')}
            >
              Layout edit
            </Button>
            <Button
              type={editorMode === 'transcription' ? 'primary' : 'default'}
              onClick={() => setEditorMode('transcription')}
            >
              Transcription edit
            </Button>
            <Button
              type={drawMode === 'rectangle' ? 'primary' : 'default'}
              onClick={() => pickDrawMode('rectangle')}
              disabled={editorMode !== 'layout'}
            >
              Rectangle segment
            </Button>
            <Button
              type={drawMode === 'polygon' ? 'primary' : 'default'}
              onClick={() => pickDrawMode('polygon')}
              disabled={editorMode !== 'layout'}
            >
              Polygon segment
            </Button>
            <Typography.Text style={{ color: '#c5ccd6' }}>
              {lines.length} {lines.length === 1 ? 'Segment' : 'Segments'}
            </Typography.Text>
            {selectedSegmentId && (
              <>
                <Button
                  disabled={editorMode !== 'layout'}
                  onClick={() => void moveSelectedSegmentRight()}
                >
                  Move segment right
                </Button>
                <Button
                  danger
                  disabled={editorMode !== 'layout'}
                  onClick={() => void deleteSelectedSegment()}
                >
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
          {transcriptionSaveMessage && (
            <Alert type="success" showIcon message={transcriptionSaveMessage} />
          )}
          {copyMessage && <Alert type="success" showIcon message={copyMessage} />}
          {mutationError && <Alert type="error" showIcon message={mutationError} />}
          {pairingError && <Alert type="warning" showIcon message={pairingError} />}
          {reviewError && <Alert type="warning" showIcon message={reviewError} />}

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
                const selected = lines.find((line) => line.id === lineId);
                setSelectedSegmentId(lineId);
                setSelectedLineId(null);
                setSaveMessage(null);
                setTranscriptionSaveMessage(null);
                setApprovedTextDraft(
                  selected ? lineTextForLayer(selected, selectedTranscriptionLayerId) : '',
                );
              }}
            />
          </div>

          {editorMode === 'transcription' && (
            <Card
              title="Transcription edit"
              style={{ background: '#101318', borderColor: '#3b4350' }}
              styles={{ header: { color: '#f7f2e8' }, body: { display: 'grid', gap: 12 } }}
            >
              <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
                Transcription layer
                <select
                  aria-label="Transcription layer"
                  value={selectedTranscriptionLayerId ?? ''}
                  onChange={selectTranscriptionLayer}
                >
                  {transcriptionLayers.map((layer) => (
                    <option key={layer.id} value={layer.id}>
                      {layer.name}
                      {layer.kind === 'model' ? ' (read-only)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <Typography.Text style={{ color: '#c5ccd6' }}>
                {selectedSegmentNumber
                  ? `Selected Segment ${selectedSegmentNumber}`
                  : 'Select a Segment to view transcription text.'}
              </Typography.Text>
              {selectedSegment && selectedTranscriptionLayer?.kind === 'ground_truth' && (
                <>
                  <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
                    Ground truth text for selected Segment
                    <Input.TextArea
                      aria-label="Ground truth text for selected Segment"
                      value={approvedTextDraft}
                      rows={3}
                      onChange={(event) => setApprovedTextDraft(event.target.value)}
                    />
                  </label>
                  <Button type="primary" onClick={() => void saveGroundTruthText()}>
                    Save Ground truth text
                  </Button>
                </>
              )}
              {selectedSegment && selectedTranscriptionLayer?.kind === 'model' && (
                <>
                  <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
                    Read-only text for selected Segment
                    <Input.TextArea
                      aria-label="Read-only text for selected Segment"
                      value={lineTextForLayer(selectedSegment, selectedTranscriptionLayer.id)}
                      rows={3}
                      readOnly
                    />
                  </label>
                  <Space wrap>
                    <Button
                      type="primary"
                      onClick={() => void copySelectedLayerToGroundTruth([selectedSegment.id])}
                    >
                      Copy selected Segment to Ground truth
                    </Button>
                    <Button onClick={() => void copySelectedLayerToGroundTruth(null)}>
                      Copy whole Page to Ground truth
                    </Button>
                  </Space>
                </>
              )}
            </Card>
          )}

          <Card
            title="Pairing"
            style={{ background: '#101318', borderColor: '#3b4350' }}
            styles={{ header: { color: '#f7f2e8' }, body: { display: 'grid', gap: 12 } }}
          >
            <Space wrap>
              <Typography.Text style={{ color: '#f7f2e8' }}>
                Review status: {part.reviewed ? 'Reviewed' : 'Unreviewed'}
              </Typography.Text>
              <Button onClick={() => void updateReviewStatus(!part.reviewed)}>
                {part.reviewed ? 'Mark unreviewed' : 'Mark reviewed'}
              </Button>
            </Space>
            <Typography.Text style={{ color: '#f7f2e8' }}>
              Pairing progress: {pairingProgress.paired_lines}/{pairingProgress.total_lines}{' '}
              Lines paired
            </Typography.Text>
            <Typography.Text style={{ color: '#c5ccd6' }}>
              Select a Segment first, then pair a candidate Text line or type approved text.
            </Typography.Text>
            <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
              Page transcription text
              <Input.TextArea
                aria-label="Page transcription text"
                value={pageTranscriptionText}
                rows={4}
                onChange={(event) => setPageTranscriptionText(event.target.value)}
              />
            </label>
            <Button onClick={() => void importPageTranscription()}>
              Import page transcription
            </Button>
            <div style={{ display: 'grid', gap: 8 }}>
              {textLines.map((textLine) => {
                const pairedIndex = textLine.paired_line_id
                  ? [...lines]
                      .sort((a, b) => a.order - b.order)
                      .findIndex((line) => line.id === textLine.paired_line_id)
                  : -1;
                const pairedLabel =
                  pairedIndex >= 0 ? ` · paired with Segment ${pairedIndex + 1}` : '';
                return (
                  <div
                    key={textLine.order}
                    style={{
                      border: '1px solid #3b4350',
                      borderRadius: 6,
                      padding: 8,
                      color: '#f7f2e8',
                    }}
                  >
                    <Typography.Text style={{ color: '#d8c7a1' }}>
                      Text line {textLine.order + 1}
                      {pairedLabel}
                    </Typography.Text>
                    <Typography.Paragraph style={{ color: '#f7f2e8', marginBottom: 8 }}>
                      {textLine.text}
                    </Typography.Paragraph>
                    <Button
                      disabled={!selectedSegmentId}
                      onClick={() => void pairTextLine(textLine.order)}
                    >
                      Pair Text line {textLine.order + 1}
                    </Button>
                  </div>
                );
              })}
              {selectedSegmentNumber && (
                <div style={{ display: 'grid', gap: 8 }}>
                  <Typography.Text style={{ color: '#c5ccd6' }}>
                    Selected Segment {selectedSegmentNumber}
                  </Typography.Text>
                  <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
                    Approved text for selected Segment
                    <Input.TextArea
                      aria-label="Approved text for selected Segment"
                      value={approvedTextDraft}
                      rows={3}
                      onChange={(event) => setApprovedTextDraft(event.target.value)}
                    />
                  </label>
                  <Button type="primary" onClick={() => void saveApprovedText()}>
                    Save approved text
                  </Button>
                </div>
              )}
            </div>
          </Card>
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
