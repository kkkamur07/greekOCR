import { describe, expect, it } from "vitest";

import { exportFileStem } from "./exportFilename";

describe("exportFileStem", () => {
  it.each([
    ["My Codex", 3, "My_Codex_page_3"],
    [
      'slash/back\\colon:star*q?quote"lt<gt>pipe|',
      2,
      "slashbackcolonstarqquoteltgtpipe_page_2",
    ],
    ["Σιναϊτικός κώδικας", 12, "Σιναϊτικός_κώδικας_page_12"],
    ["", 1, "document_page_1"],
    ["trailing dots...", 4, "trailing_dots_page_4"],
  ])("names %j page %i as %s", (name, page, expected) => {
    expect(exportFileStem(name, page)).toBe(expected);
  });

  it("caps long titles", () => {
    expect(exportFileStem("x".repeat(500), 7)).toBe(`${"x".repeat(80)}_page_7`);
  });
});
