# Browsary OpenAI Provider

A ChatGPT-powered browser-automation agent that analyzes page structure and generates JSON-driven automation pipelines. Built on top of OpenAI’s API, Puppeteer-core, and the Browsary pipeline framework.

## Installation

```bash
npm install @silyze/browsary-openai-provider
```

## Quick Start

```ts
import OpenAI from "openai";
import { OpenAiProvider, OpenAiConfig } from "@silyze/browsary-openai-provider";
import { PipelineProvider } from "@silyze/browsary-pipeline";
import { createConsoleLogger } from "@silyze/logger";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pipelineProvider = new PipelineProvider();
const logger = createConsoleLogger();

const config: OpenAiConfig = {
  openAi: openai,
  pipelineProvider,
  logger,
  models: {
    analyze: process.env.ANALYZE_MODEL,
    generate: process.env.GENERATE_MODEL,
  },
};

const aiProvider = new OpenAiProvider(config, async (page, fnName, params) => {
  // implement browser function calls (e.g., puppeteer page[fnName](...))
  // return JSON-serializable result
});

const page = await browser.newPage();
const userPrompt = "Search for 'open source licenses' and list links";
const previousPipeline = {};

const analysis = await aiProvider.analyze(page, userPrompt, previousPipeline);
if (analysis.result) {
  const generation = await aiProvider.generate(
    page,
    analysis.result,
    previousPipeline
  );
  console.log("Generated pipeline:", generation.result);
}
```

## Configuration

### `OpenAiConfig`

| Property           | Type                                      | Description                                           |
| ------------------ | ----------------------------------------- | ----------------------------------------------------- |
| `openAi`           | `OpenAI \| Promise<OpenAI>`               | An instantiated OpenAI client (or a promise thereof). |
| `pipelineProvider` | `PipelineProvider`                        | Compiles and runs generated pipelines.                |
| `logger`           | `Logger`                                  | Scoped logger for debug/info/error events.            |
| `models?`          | `{ analyze?: string; generate?: string }` | Override default OpenAI model identifiers.            |

## API Reference

### Class: `OpenAiProvider`

Extends `AiProvider<Page, OpenAiConfig>`. Handles two phases: analysis and generation.

#### `constructor(config: OpenAiConfig, functionCall: (page: Page, name: string, params: any) => Promise<unknown>)`

- **config**: See Configuration.
- **functionCall**: Invoked on ChatGPT function calls to map to Puppeteer actions.

#### `analyze(page: Page, userPrompt: string, previousPipeline: Record<string, GenericNode>, onMessages?: (messages: unknown[]) => void): Promise<AiResult<AnalysisResult>>`

Performs the analysis phase:

1. Logs start of analysis.
2. Sends system & user prompts to OpenAI, using `analyzeTools` and `analyzePrompt`.
3. Follows up on function calls (`querySelector`, `goto`, etc.) via `functionCall`.
4. Returns `{ result?: { analysis, prompt }, messages }`.

#### `generate(page: Page, analysisResult: AnalysisResult, previousPipeline: Record<string, GenericNode>, onMessages?: (messages: unknown[]) => void): Promise<AiResult<Pipeline>>`

Performs the generation phase:

1. Sends system & user prompts using `pipelinePrompt(analysis)`.
2. Emits JSON pipeline via `pipelineOutput`.
3. Retries up to `MAX_RESPONSE_FIX_RETRY` on parse or schema validation errors.
4. Returns `{ result?: Pipeline, messages }` on success.

## Prompts & Schemas

### Analysis Phase

- **`analyzeTools`**: Definitions for OpenAI function calls:

  - `querySelector`
  - `querySelectorAll`
  - `goto`
  - `click`
  - `type`
  - `url`

- **`analyzePrompt`**: System prompt guiding the model to explore page structure, identify selectors, and avoid executing user actions.

- **`analyzeOutputSchema`**:

  ```json
  {
    "type": "object",
    "properties": {
      "selectors": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "selector": { "type": "string" },
            "locatedAtUrl": { "type": "string" },
            "description": { "type": "string" },
            "type": { "enum": ["guess", "tested-valid", "tested-fail"] }
          },
          "required": ["selector", "locatedAtUrl", "description", "type"]
        }
      },
      "metadata": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["selectors", "metadata"]
  }
  ```

### Generation Phase

- **`pipelinePrompt(analysis)`**: System prompt for converting analysis output into a JSON pipeline, enforcing `pipelineSchema` compliance and prohibiting extra text.

- **`pipelineOutput`**: Instructs OpenAI to emit JSON matching `genericNodeSchema`.

- **Schema validation helpers**:

  - `validatePipelineSchema(obj): boolean`
  - `getPipelineValidationErrors(): Ajv.ErrorObject[]`

## Types

- **`OpenAiConfig`**
- **`AnalysisResult`**: `{ selectors: …; metadata: …; prompt: string }`
- **`Pipeline`**: Mapping of step names to node definitions with inputs/outputs/dependsOn
- **`AiResult<T>`**: `{ result?: T; messages: unknown[] }`

Refer to the TypeScript definitions for complete details.

## Environment Variables

- `OPENAI_API_KEY` — Your API key for OpenAI.
- `OPENAI_DEFAULT_MODEL` — Default model for both phases.
- `OPENAI_DEFAULT_ANALYZE_MODEL` — Override model for analysis.
- `OPENAI_DEFAULT_GENERATE_MODEL` — Override model for generation.
