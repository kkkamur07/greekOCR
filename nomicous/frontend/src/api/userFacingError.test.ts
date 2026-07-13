import { describe, expect, it } from "vitest";

import { ApiError } from "./errors";
import { errorRef, userFacingMessage } from "./userFacingError";

describe("userFacingMessage", () => {
  it("prefers allowlisted API messages", () => {
    expect(
      userFacingMessage(new ApiError("Access denied", 403), "fallback"),
    ).toBe("Access denied");
  });

  it("hides stack-like noise behind a status fallback", () => {
    expect(
      userFacingMessage(
        new ApiError(
          'Traceback (most recent call last):\nFile "app.py", line 1',
          500,
        ),
      ),
    ).toMatch(/something went wrong on the server/i);
  });

  it("exposes correlation refs from ApiError", () => {
    const err = new ApiError("Request failed", 500, "abc123");
    expect(errorRef(err)).toBe("abc123");
  });
});
