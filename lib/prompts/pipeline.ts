import {
  pipelineSchema,
  genericNodeSchema,
  RefType,
} from "@silyze/browsary-pipeline";
import { AnalyzeOutput } from "@silyze/browsary-ai-provider";

import type OpenAI from "openai";
import Ajv from "ajv";

type InputSchema =
  (typeof pipelineSchema)["additionalProperties"]["anyOf"][number]["properties"]["inputs"]["properties"][string];
type OutputSchema =
  (typeof pipelineSchema)["additionalProperties"]["anyOf"][number]["properties"]["outputs"]["properties"][string];

function getType(schema: InputSchema | OutputSchema) {
  if ("properties" in schema) {
    return schema.properties.type[RefType];
  }
  if ("anyOf" in schema) {
    const first = schema.anyOf[0];
    if ("properties" in first) {
      return first.properties.type[RefType];
    }
  }

  return "any";
}

export function formatPipelineNodes(schema: typeof pipelineSchema): string {
  const nodes = schema.additionalProperties.anyOf;
  return nodes
    .map((node) => {
      const name = node.properties.node.const;
      const description = node.description ?? "";
      const inputs = Object.entries(node.properties.inputs?.properties ?? {});
      const outputs = Object.entries(node.properties.outputs?.properties ?? {});
      const io = [
        inputs.length
          ? `**Inputs**: \`{ ${inputs
              .map(([key, value]) => `${key}: ${getType(value)}`)
              .join(", ")} }\``
          : null,
        outputs.length
          ? `**Outputs**: \`{ ${outputs
              .map(([key, value]) => `${key}: ${getType(value)}`)
              .join(", ")} }\``
          : null,
      ]
        .filter(Boolean)
        .join(". ");
      return `- \`${name}\` → ${description}${io ? `. ${io}` : ""}`;
    })
    .join("\n");
}

export const pipelineOutput = {
  format: {
    type: "json_schema",
    name: "pipeline",
    schema: genericNodeSchema,
    strict: false,
  },
} as const;
const ajv = new Ajv();
const validate = ajv.compile(pipelineSchema);

export function getPipelineValidationErrors() {
  return validate.errors;
}

export function validatePipelineSchema(pipeline: unknown) {
  return validate(pipeline);
}

export const PIPELINE_GENERATE_MODEL = (process.env
  .OPENAI_DEFAULT_GENERATE_MODEL ??
  process.env.OPENAI_DEFAULT_MODEL) as OpenAI.ResponsesModel;

const pipelineNodeList = `## Available pipeline nodes:\n${formatPipelineNodes(
  pipelineSchema
)}`;

const pipelinePrompt = (analysis: AnalyzeOutput) => `
You are a browser automation agent. Your task is to convert natural language instructions into a JSON pipeline that automates interactions with a web browser.

# Analyze phase output:
\`\`\`
${JSON.stringify(analysis, null, 2)}
\`\`\`

Now that the analysis phase is complete, proceed with the **Generation phase**:

- Create a JSON object representing the pipeline that performs the user-requested browser automation task.
- Each step (e.g., navigating, clicking, typing, comparing, logging) must be represented as a distinct node.
- Use only the available node types listed below (you cannot invent new ones).
- Follow the provided JSON schema strictly.
- Use "outputOf" references to connect outputs from earlier nodes to inputs in later nodes.
- **Before emitting any node, use the \`getNodeSchema(nodeType)\` function to verify the required structure, inputs, and outputs are valid.**

## Pipeline format

The pipeline must be a single JSON object. Each key is a unique, descriptive step name (e.g., \`"goto_login"\`, \`"click_search"\`). Each step must define:

- \`node\`: the node type.
- \`inputs\`: all required inputs.
- \`outputs\`: any values produced.
- \`dependsOn\`: required step(s) that must run before this one.

## Advanced Concepts

### 1. Conditional Execution

You can conditionally execute steps using conditional \`dependsOn\` values:

\`\`\`json
"dependsOn": [
  {
    "nodeName": "check_condition",
    "outputName": "result"
  }
]
\`\`\`

This step will only run if the output referenced above evaluates to a truthy value.

### 2. Looping

To build loops:
- Declare a mutable variable (e.g., with \`declare::number\`).
- Use conditional logic (e.g., \`logic::greaterThan\`) to control flow.
- Use conditional \`dependsOn\` to gate each iteration.
- Redirect outputs to update state.

### 3. Output Redirection

You can overwrite a previous step's output using this pattern:

\`\`\`json
"outputs": {
  "result": {
    "nodeName": "counter",
    "outputName": "value"
  }
}
\`\`\`

This enables mutable state for counters, accumulators, etc.

### 4. Schema Validation (Important)

Before emitting any node, you must:
- Call \`getNodeSchema(nodeType)\`
- Use its structure to ensure your \`inputs\`, \`outputs\`, and shape match what the node expects
- Do not guess — validation against this schema is mandatory

${pipelineNodeList}

---

## Constraints

- Return a valid JSON object that strictly conforms to the pipeline schema.
- Never assume an initial page state. Always navigate first.
- Do not simulate anchor navigation by clicking on links after a goto. Use \`page::goto\` with the correct full URL directly.
- Always use a \`create_page\` output when referencing a \`page\`.
- All outputs must be correctly referenced from the node that produces them.
- The response must be JSON only. Do not include any extra text, comments, or markdown.

---

## Example: Looping with Condition and Redirection

\`\`\`json
{
  "counter": {
    "node": "declare::number",
    "inputs": {
      "value": { "type": "constant", "value": 10 }
    },
    "outputs": {
      "value": "value2"
    },
    "dependsOn": []
  },
  "loop": {
    "node": "log::info",
    "inputs": {
      "value": { "type": "constant", "value": "Test" }
    },
    "outputs": {},
    "dependsOn": [
      {
        "nodeName": "check",
        "outputName": "result2"
      }
    ]
  },
  "decrement_counter": {
    "node": "logic::subtract",
    "inputs": {
      "a": {
        "type": "outputOf",
        "nodeName": "counter",
        "outputName": "value2"
      },
      "b": { "type": "constant", "value": 1 }
    },
    "outputs": {
      "result": {
        "nodeName": "counter",
        "outputName": "value2"
      }
    },
    "dependsOn": [ "loop" ]
  },
  "check": {
    "node": "logic::greaterThan",
    "inputs": {
      "a": {
        "type": "outputOf",
        "nodeName": "counter",
        "outputName": "value2"
      },
      "b": { "type": "constant", "value": 0 }
    },
    "outputs": {
      "result": "result2"
    },
    "dependsOn": [
      "counter",
      "decrement_counter"
    ]
  }
}
\`\`\`

This demonstrates conditional looping using output redirection and evaluation-based gating.
`;

export default pipelinePrompt;
