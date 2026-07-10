import { useRef, useState, type ReactNode } from "react";
import {
  TransformComponent,
  TransformWrapper,
  type ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch";

type PublicZoomSurfaceProps = {
  children: ReactNode;
  ariaLabel?: string;
};

const ZOOM_STEP = 0.15;
const ZOOM_ANIMATION_MS = 200;

export function PublicZoomSurface({
  children,
  ariaLabel,
}: PublicZoomSurfaceProps) {
  const transformRef = useRef<ReactZoomPanPinchRef>(null!);
  const [zoomLevel, setZoomLevel] = useState(1);

  return (
    <div className="pub-zoom-surface" aria-label={ariaLabel}>
      <TransformWrapper
        ref={transformRef}
        initialScale={1}
        minScale={0.2}
        maxScale={16}
        centerOnInit
        limitToBounds={false}
        wheel={{ step: ZOOM_STEP, smoothStep: 0.012 }}
        pinch={{ step: 6 }}
        doubleClick={{
          disabled: false,
          step: 0.75,
          mode: "zoomIn",
          animationTime: ZOOM_ANIMATION_MS,
        }}
        panning={{ velocityDisabled: false, wheelPanning: false }}
        zoomAnimation={{
          disabled: false,
          size: ZOOM_STEP,
          animationTime: ZOOM_ANIMATION_MS,
        }}
        velocityAnimation={{
          disabled: false,
          sensitivity: 1,
          animationTime: 350,
        }}
        onTransformed={(ref) => setZoomLevel(ref.state.scale)}
      >
        {() => (
          <>
            <div className="pub-zoom-controls" aria-label="Zoom controls">
              <span className="pub-zoom-controls__level">
                {Math.round(zoomLevel * 100)}%
              </span>
              <button
                type="button"
                className="pub-zoom-controls__btn"
                title="Zoom in"
                aria-label="Zoom in"
                onClick={() =>
                  transformRef.current?.zoomIn(ZOOM_STEP, ZOOM_ANIMATION_MS)
                }
              >
                +
              </button>
              <button
                type="button"
                className="pub-zoom-controls__btn"
                title="Zoom out"
                aria-label="Zoom out"
                onClick={() =>
                  transformRef.current?.zoomOut(ZOOM_STEP, ZOOM_ANIMATION_MS)
                }
              >
                −
              </button>
              <button
                type="button"
                className="pub-zoom-controls__btn"
                title="Fit to view"
                aria-label="Fit to view"
                onClick={() =>
                  transformRef.current?.centerView(1, ZOOM_ANIMATION_MS)
                }
              >
                ⊡
              </button>
              <button
                type="button"
                className="pub-zoom-controls__btn"
                title="Reset zoom"
                aria-label="Reset zoom"
                onClick={() =>
                  transformRef.current?.resetTransform(ZOOM_ANIMATION_MS)
                }
              >
                ⟲
              </button>
            </div>
            <p className="pub-zoom-hint">
              Scroll to zoom · drag to pan · double-click to zoom in
            </p>
            <TransformComponent
              wrapperClass="pub-zoom-surface__wrapper"
              contentClass="pub-zoom-surface__content"
            >
              {children}
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
