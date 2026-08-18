/**
 * The platform's own refusal copy, read from the platform.
 *
 * The frontend does not own this sentence and must never re-word it: when no
 * **inference host** has **capacity** the platform answers 409 with a message
 * naming the situation, and the interface's whole job is to put that message
 * somewhere the researcher can act on. Copying the string into TypeScript would
 * create a second source of truth that drifts silently.
 *
 * Tests use this to drive the refusal path with the real sentence rather than
 * an invented one. It reads the file at call time, so it is Node-only and has
 * no place in a browser bundle - nothing under `src/` imports it outside tests.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const EXECUTION_DOMAIN_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../backend/ml/domain/execution.py",
);

function pythonConstant(source: string, name: string): string {
  // `NAME = (\n  "part one "\n  "part two"\n)` - implicit concatenation of
  // adjacent string literals, which is how the platform writes long messages.
  const declaration = new RegExp(`^${name} = \\(([\\s\\S]*?)\\n\\)`, "m").exec(
    source,
  );
  const body = declaration
    ? declaration[1]
    : new RegExp(`^${name} = (.*)$`, "m").exec(source)?.[1];
  if (!body) {
    throw new Error(`${name} is not defined in ${EXECUTION_DOMAIN_PATH}`);
  }
  const parts = body.match(/"([^"]*)"/g);
  if (!parts) {
    throw new Error(`${name} in ${EXECUTION_DOMAIN_PATH} is not a string`);
  }
  return parts.map((part) => part.slice(1, -1)).join("");
}

/** What the platform says when neither inference host can take the work. */
export function platformNoCapacityMessage(): string {
  return pythonConstant(
    readFileSync(EXECUTION_DOMAIN_PATH, "utf8"),
    "NO_CAPACITY_MESSAGE",
  );
}
