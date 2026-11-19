import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { selfCheck } from "./transfer.ts";

describe("universal lora bridge", () => {
  for (const check of selfCheck()) {
    it(check.name, () => {
      assert.equal(check.pass, true, check.detail);
    });
  }
});
