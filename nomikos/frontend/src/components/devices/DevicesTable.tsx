import { Popconfirm } from "antd";
import type { DeviceResponse, DeviceStatus } from "../../api/resources";

type DevicesTableProps = {
  id: string;
  caption: string;
  devices: DeviceResponse[];
  loading: boolean;
  emptyText: string;
  onRevoke: (deviceId: string) => void;
  revokingDeviceId: string | null;
};

const STATUS_LABEL: Record<DeviceStatus, string> = {
  pairing: "pairing",
  online: "online",
  idle: "idle",
  offline: "offline",
  revoked: "revoked",
};

/**
 * `badge-live` is the only green badge the sheet defines and `badge-archived`
 * the only muted one, so the five device states borrow them rather than
 * inventing classes the stylesheet does not have.
 */
function statusBadgeClass(status: DeviceStatus): string {
  if (status === "online") return "badge badge-live";
  if (status === "revoked") return "badge badge-archived";
  return "badge badge-draft";
}

function formatWhen(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** A device that has never called in has no `last_seen_at` at all. */
function formatLastSeen(device: DeviceResponse): string {
  if (device.status === "revoked") {
    return device.revoked_at
      ? `revoked ${formatWhen(device.revoked_at)}`
      : "revoked";
  }
  return device.last_seen_at ? formatWhen(device.last_seen_at) : "never";
}

export function DevicesTable({
  id,
  caption,
  devices,
  loading,
  emptyText,
  onRevoke,
  revokingDeviceId,
}: DevicesTableProps) {
  return (
    <div className="data-panel mb-4">
      <table className="data-list" aria-labelledby={id}>
        <thead>
          <tr>
            <th scope="col">Device</th>
            <th scope="col">Status</th>
            <th scope="col">Last seen</th>
            <th scope="col">Token</th>
            <th scope="col">
              <span className="text-muted">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr className="data-list-empty">
              <td colSpan={5}>Loading…</td>
            </tr>
          ) : devices.length === 0 ? (
            <tr className="data-list-empty">
              <td colSpan={5}>{emptyText}</td>
            </tr>
          ) : (
            devices.map((device) => {
              const revoked = device.status === "revoked";
              return (
                <tr key={device.id}>
                  <td>
                    <span className="row-title">{device.name}</span>
                    <span className="row-sub">
                      {device.platform} · helper {device.helper_version} ·
                      paired {formatWhen(device.paired_at)}
                    </span>
                  </td>
                  <td>
                    <span className={statusBadgeClass(device.status)}>
                      {STATUS_LABEL[device.status]}
                    </span>
                  </td>
                  <td className="col-muted">
                    {formatLastSeen(device)}
                    {device.last_seen_ip && !revoked ? (
                      <span className="row-sub">{device.last_seen_ip}</span>
                    ) : null}
                  </td>
                  <td className="col-muted">
                    <code>{device.token_prefix}…</code>
                  </td>
                  <td className="col-action">
                    <div className="data-list-actions">
                      {revoked ? (
                        <span className="text-muted text-sm">—</span>
                      ) : (
                        <Popconfirm
                          title={`Revoke ${device.name}?`}
                          description="That computer stops being able to run your jobs on its next request. Pairing it again means starting over from its terminal."
                          okText="Revoke"
                          cancelText="Keep"
                          okButtonProps={{
                            danger: true,
                            loading: revokingDeviceId === device.id,
                          }}
                          onConfirm={() => onRevoke(device.id)}
                        >
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm btn--danger-ghost"
                            disabled={revokingDeviceId === device.id}
                            aria-label={`Revoke device ${device.name}`}
                          >
                            Revoke
                          </button>
                        </Popconfirm>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
      <span className="visually-hidden">{caption}</span>
    </div>
  );
}
