import test from "node:test";
import assert from "node:assert/strict";
import Ajv from "ajv";

import { createPipelineToolSchema } from "../lib/schema-tools";

test("pipeline tool schema compiles with Ajv", () => {
  const schema = createPipelineToolSchema();
  const ajv = new Ajv({ strict: false });

  assert.doesNotThrow(() => {
    ajv.compile(schema);
  });

  assert.ok(schema.$defs?.GenericNode, "GenericNode definition is missing");
});
