/**
 * `local_only` is gone from the interface, and so is the claim that justified
 * it.
 *
 * ADR 0002 retired the mode because its headline justification - manuscripts
 * never leave the machine - was never true: page images live in the platform's
 * media store and the browser downloads them from there today. Deleting the
 * control without deleting the claim would leave the falsehood behind, so both
 * are asserted here, over the real files rather than a summary of them.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..");
const FRONTEND = resolve(SRC, "..");

function sourceFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((entry) => {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.(ts|tsx)$/.test(path) ? [path] : [];
  });
}

/**
 * Comments are stripped before scanning, and tests are left out entirely.
 * What is asserted is the interface itself - the code that runs and the copy a
 * researcher reads - not the prose explaining why the mode is gone, which is
 * exactly what a record of a deletion should be allowed to say.
 */
function interfaceSource(path: string): string {
  return readFileSync(path, "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[^\n"'`]*\/\/.*$/gm, "");
}

describe("local_only is retired from the interface", () => {
  const files = sourceFiles(SRC).filter((path) => !/\.test\.tsx?$/.test(path));

  it("has source files to check", () => {
    expect(files.length).toBeGreaterThan(100);
  });

  it("names no local-only execution mode anywhere in the interface", () => {
    const offenders = files.filter((path) =>
      /local_only|localOnly|local-only|Local only/i.test(interfaceSource(path)),
    );

    expect(offenders).toEqual([]);
  });

  it("makes no claim that nothing is sent to the cloud", () => {
    const offenders = files.filter((path) =>
      /nothing is (?:ever )?sent to the cloud|never leaves? (?:your|this) (?:computer|machine)/i.test(
        interfaceSource(path),
      ),
    );

    expect(offenders).toEqual([]);
  });

  it("does not appear in the API contract the interface is generated from", () => {
    for (const contract of ["openapi/openapi.json", "src/api/schema.d.ts"]) {
      expect(readFileSync(resolve(FRONTEND, contract), "utf8")).not.toContain(
        "local_only",
      );
    }
  });

  it("offers no per-job execution target control", () => {
    // The account setting is the only input. A request body field named for an
    // execution target would be a per-job toggle by another name.
    for (const request of ["SegmentPartRequest", "TranscribePartRequest"]) {
      const schema = JSON.parse(
        readFileSync(resolve(FRONTEND, "openapi/openapi.json"), "utf8"),
      ).components.schemas[request];
      expect(Object.keys(schema.properties ?? {})).toEqual(
        expect.not.arrayContaining([
          expect.stringMatching(/^(execution|prefer)/),
        ]),
      );
    }
  });
});
