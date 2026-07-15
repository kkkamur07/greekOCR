import { Spin } from "antd";

type ContentRegionLoadingProps = {
  label: string;
};

/** Spinner for a waiting content region; surrounding chrome stays mounted. */
export function ContentRegionLoading({ label }: ContentRegionLoadingProps) {
  return (
    <div
      className="content-region-loading"
      role="status"
      aria-busy="true"
      aria-label={label}
    >
      <Spin />
    </div>
  );
}
