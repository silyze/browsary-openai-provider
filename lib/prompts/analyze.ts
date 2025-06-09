import OpenAI from "openai";
import {
  booleanInputType,
  integerInputType,
  stringInputType,
  waitEventType,
} from "@silyze/browsary-pipeline";

export const analyzeTools: OpenAI.Responses.FunctionTool[] = [
  {
    type: "function",
    name: "querySelector",
    description: "Query a single HTML element from the current page as JSON",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        selector: {
          type: "string",
        },
      },
      required: ["selector"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "querySelectorAll",
    description: "Query multiple HTML elements from the current page as JSON",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        selector: {
          type: "string",
        },
      },
      required: ["selector"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "goto",
    description: "Navigate to a URL",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        url: {
          type: "string",
        },
        waitUntil: waitEventType,
      },
      required: ["url", "waitUntil"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "click",
    description: "Click on a HTML element with a selector",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        selector: {
          type: "string",
        },
        waitForNavigation: booleanInputType,
      },
      required: ["selector", "waitForNavigation"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "type",
    description: "Type in a HTML element with a selector",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        selector: {
          type: "string",
        },
        text: stringInputType,
        delayMs: integerInputType,
      },
      required: ["selector", "text", "delayMs"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "url",
    description: "Get the current URL of the page",
    strict: true,
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false,
    },
  },
];

export const ANALYZE_MODEL = (process.env.OPENAI_DEFAULT_ANALYZE_MODEL ??
  process.env.OPENAI_DEFAULT_MODEL) as OpenAI.ResponsesModel;

export const analyzeOutputSchema = {
  type: "object",
  properties: {
    metadata: {
      description:
        "This array should include information about non-obvious part of the analysis",
      type: "array",
      items: {
        type: "string",
      },
    },
    selectors: {
      type: "array",
      items: {
        type: "object",
        properties: {
          selector: {
            type: "string",
          },
          locatedAtUrl: {
            type: "string",
          },
          description: {
            type: "string",
          },
          type: {
            anyOf: [
              {
                type: "string",
                const: "guess",
              },
              {
                type: "string",
                const: "tested-valid",
              },
              {
                type: "string",
                const: "tested-fail",
              },
            ],
          },
        },
        required: ["selector", "locatedAtUrl", "description", "type"],
        additionalProperties: false,
      },
    },
  },
  required: ["selectors", "metadata"],
  additionalProperties: false,
};

const analyzePrompt = `
You are a browser automation analysis tool.

Your goal is for a given prompt to:
- Analyze the structure of the page.
- Find relevant DOM selectors and the page URL they are located at.
- If a selector is not found on the current page, you are supposed to navigate to a more specific page.

Rules:
- Prefer to call "querySelectorAll" on "body" instead of querying for specific selectors AS IT RETURNS THE WHOLE NODE TREE WITH ALL IT'S DESCENDANTS.
- Only call the "querySelector" or "querySelectorAll" functions after a navigation, form submit, button click. DO NOT CALL IT WHEN THE BODY ISN'T EXPECTED TO HAVE CHANGED.
- Only output selectors that EXIST and are entirely relevant to the prompt's objective.
- You MUST NOT preform the action requested by the prompt, YOU ARE ONLY SUPPOSED TO ANALYZE.
- You MUST NOT attempt to generate the pipeline itself.

When all the steps and selectors are known to generate the pipeline, return the output as soon as possible.

`;
export default analyzePrompt;
