import { pipelineSchema, genericNodeSchema } from "@silyze/browsary-pipeline";
import { AnalyzeOutput } from "@silyze/browsary-ai-provider";

import type OpenAI from "openai";
import Ajv from "ajv";

export function formatPipelineNodes(schema: typeof pipelineSchema): string {
  const nodes = schema.additionalProperties.anyOf;
  return nodes
    .map((node) => {
      const name = node.properties.node.const;
      const description = node.description ?? "";
      const inputs = Object.keys(node.properties.inputs?.properties ?? {});
      const outputs = Object.keys(node.properties.outputs?.properties ?? []);
      const io = [
        inputs.length ? `Inputs: ${inputs.join(", ")}` : null,
        outputs.length ? `Outputs: ${outputs.join(", ")}` : null,
      ]
        .filter(Boolean)
        .join(". ");
      return `- ${name} → ${description}${io ? `. ${io}` : ""}`;
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

# Analyize phase output:
${JSON.stringify(analysis)}


Now that the analysis phase is complete, proceed with the **Generation phase**:

- Create a JSON object representing the pipeline that performs the user-requested browser automation task.
- Each step (e.g., navigating, clicking, typing) must be represented as a distinct node.
- Use only the available node types listed below (you cannot invent new ones).
- Follow the provided JSON schema strictly.
- Use "outputOf" references to connect outputs from earlier nodes to inputs in later nodes.

## Pipeline format:

The pipeline must be a single JSON object, where each key is a unique, descriptive step name (e.g., \`"goto_login"\` or \`"click_search"\`). Each step must define:

- The \`node\` type (one of the allowed pipeline nodes).
- Any \`inputs\` required by the node.
- Any \`outputs\` the node produces.
- A \`dependsOn\` field specifying prior steps this step depends on.

Use \`"outputOf"\` references to pass outputs between steps.

${pipelineNodeList}

# Constraints
- You must return a valid JSON object that strictly conforms to the pipelineSchema. Any deviation is unacceptable.
- Always navigate to the correct page before interacting with any selectors. Do not assume the initial page is correct.
- When handling navigation via <a href>, you are forbidden from using page::goto followed by page::click. You must use page::goto with the full destination URL instead.
- All references to the page must come from a valid \`create_page\` output.
- All references to an output must come from the correct originating node. Do not reference an output from a node that doesn’t produce it.
- Your response must consist of the JSON output only. Do not include explanations, markdown, or any additional text under any circumstances.

---

## Bad Example (Violates Constraints)

\`\`\`json
{
  "goto_home": {
    "node": "page::goto",
    "dependsOn": "create_page",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "url": {
        "type": "constant",
        "value": "https://www.example.com"
      }
    },
    "outputs": {}
  },
  "click_category_link": {
    "node": "page::click",
    "dependsOn": "goto_home",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "selector": {
        "type": "constant",
        "value": "a[href='/category/search']"
      }
    },
    "outputs": {}
  },
  "type_search_query": {
    "node": "page::type",
    "dependsOn": "click_category_link",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "selector": {
        "type": "constant",
        "value": "input#search"
      },
      "text": {
        "type": "constant",
        "value": "Search Query"
      },
      "delayMs": {
        "type": "constant",
        "value": 100
      }
    },
    "outputs": {}
  }
}
\`\`\`

** Why it's wrong:**
- It navigates with \`page::goto\` and then clicks on a navigation link using \`page::click\`.
- This violates the rule against simulating anchor navigation — the final URL must be opened directly with \`page::goto\`.
- Also, it repeatedly references the same output without proper dependency flow.

---

## Good Example (Follows Constraints)

\`\`\`json
{
  "goto_target_page": {
    "node": "page::goto",
    "dependsOn": "create_page",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "url": {
        "type": "constant",
        "value": "https://www.example.com/category/search"
      }
    },
    "outputs": {}
  },
  "type_search_query": {
    "node": "page::type",
    "dependsOn": "goto_target_page",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "selector": {
        "type": "constant",
        "value": "input#search"
      },
      "text": {
        "type": "constant",
        "value": "Search Query"
      },
      "delayMs": {
        "type": "constant",
        "value": 100
      }
    },
    "outputs": {}
  },
  "submit_form": {
    "node": "page::click",
    "dependsOn": "type_search_query",
    "inputs": {
      "page": {
        "type": "outputOf",
        "nodeName": "create_page",
        "outputName": "page_1"
      },
      "selector": {
        "type": "constant",
        "value": "button#submit"
      }
    },
    "outputs": {}
  }
}
\`\`\`

**Why it's correct:**
- Uses \`page::goto\` to reach the actual destination page directly.
- Properly waits for context before interacting with page elements.
- References the \`page\` output from the correct node (\`create_page\`).
`;
export default pipelinePrompt;
