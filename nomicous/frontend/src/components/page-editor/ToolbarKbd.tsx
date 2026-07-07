type ToolbarKbdProps = {
  children: string;
};

export function ToolbarKbd({ children }: ToolbarKbdProps) {
  return <kbd className="pe-kbd">{children}</kbd>;
}
