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
    agent: process.env.AGENT_MODEL,
  },
};

const aiProvider = new OpenAiProvider(config, async (page, fnName, params) => {
  // implement browser function calls (e.g., puppeteer page[fnName](...))
  // return JSON-serializable result
});

const page = await browser.newPage();
const userPrompt = "Search for 'open source licenses' and list links";
let previousPipeline = {};
let conversation =
  await aiProvider.prompt(page, {
    userPrompt,
    previousPipeline,
    onPipelineUpdate: (pipeline) => {
      const json = pipeline.toJSON();
      previousPipeline = json;
      console.log("Updated pipeline", json);
    },
    onStatusUpdate: (status) => logger.log("info", "status", status),
  });

// Persist `conversation.state` somewhere durable if you need to resume later.
while (!conversation.isComplete) {
  conversation = await aiProvider.continuePrompt(page, {
    conversation,
    previousPipeline,
    onPipelineUpdate: (pipeline) => {
      const json = pipeline.toJSON();
      previousPipeline = json;
      console.log("Updated pipeline", json);
    },
    onStatusUpdate: (status) => logger.log("info", "status", status),
  });
}
```

## Example CLI Tool

The repository ships with an interactive CLI demo that showcases prompting, pausing, persisting, and resuming a Browsary + OpenAI conversation.

```bash
npm run start:example
```

The CLI will:

- ask for your high-level prompt and automatically gather clarifications if the request is vague, so the agent always has enough context;
- resume from `.browsary-cli/conversation.<hash>.json` when you restart it, letting you continue work after a pause or crash;
- stream status updates, persist every pipeline revision, and let you inject new instructions mid-run (`addMessages`);
- support `pause`/`resume` control requests so you can safely stop the agent, then continue once you're ready.

All cached prompts, conversations, and pipelines live under `.browsary-cli/` to keep the workspace tidy.

## Configuration

### `OpenAiConfig`

| Property           | Type                                      | Description                                           |
| ------------------ | ----------------------------------------- | ----------------------------------------------------- |
| `openAi`           | `OpenAI \| Promise<OpenAI>`               | An instantiated OpenAI client (or a promise thereof). |
| `pipelineProvider` | `PipelineProvider`                        | Compiles and runs generated pipelines.                |
| `logger`           | `Logger`                                  | Scoped logger for debug/info/error events.            |
| `models?`          | `{ agent?: string; generate?: string; analyze?: string }` | Override default OpenAI model identifiers (the agent model is used for the unified workflow). |

## API Reference

### Class: `OpenAiProvider`

Extends `AiProvider<Page, OpenAiConfig>`. It orchestrates a single-phase agent that can browse, reason, chat, and emit pipelines within one resumable conversation.

#### `constructor(config: OpenAiConfig, functionCall: (page: Page, name: string, params: any) => Promise<unknown>)`

- **config**: See Configuration.
- **functionCall**: Invoked on ChatGPT function calls to map to Puppeteer actions.

#### `prompt(page: Page, params: PromptParams): Promise<AiAgentConversationState<OpenAiConversationState>>`

Starts a new conversation. The provider will:

1. Expose browsing, pipeline, and communication tools simultaneously so the agent can explore, validate, emit, or simply chat in a single loop.
2. Emit `onStatusUpdate` for browsing milestones, chat messages, pending questions, output payloads, and verification steps.
3. Invoke `onPipelineUpdate` **only when** the compiled pipeline differs from `params.previousPipeline`.
4. Return a conversation state object (matching `OpenAiConversationState`) that captures agent metadata, pending questions, and pipeline history for later resumption.

#### `continuePrompt(page: Page, params: ContinuePromptParams): Promise<AiAgentConversationState<OpenAiConversationState>>`

Resumes a prior conversation. Pass the previously returned `conversation` plus the pipeline you currently have deployed. The provider will:

1. Restore the agent's pending questions, metadata, and pipeline history from `conversation.state`.
2. Clear cached artifacts when `controlRequests` inject new instructions.
3. Respect pause/resume requests:
   - `{ type: "pause", reason?: string }` stops after the current safe point.
   - `{ type: "resume" }` continues a paused conversation.
   - `{ type: "addMessages", messages: unknown[] }` appends extra user instructions (clears cached agent/pipeline data).
4. Return the updated state so you can persist it again.

`onMessages` receives the raw OpenAI message history, enabling streaming/debug UI if desired.

## Prompts & Schemas

### Unified Agent Prompt & Tooling

- **`agentPrompt(options)`**: System prompt guiding the single-phase agent. It describes how to browse, chat, compile/emit pipelines, request user input, and provide direct output data without switching phases.
- **`analyzeTools`**: Reusable browsing functions exposed to the agent:
  - `querySelector`
  - `querySelectorAll`
  - `goto`
  - `click`
  - `type`
  - `url`
- **Schema utilities** (unchanged):
  - `validatePipelineSchema(obj): boolean`
  - `getPipelineValidationErrors(): Ajv.ErrorObject[]`
  - `genericNodeSchema` / `pipelineSchema`: JSON schemas describing valid Browsary pipelines.

## Types

- **`OpenAiConfig`**
- **`OpenAiConversationState`** — serializable snapshot you should persist between `prompt`/`continuePrompt` calls.
- **`PromptParams` / `ContinuePromptParams`** — include callbacks (`onPipelineUpdate`, `onStatusUpdate?`, `onMessages?`), `previousPipeline`, optional `controlRequests`, and (for `prompt`) the initial `userPrompt`.
- **`AiAgentConversationState<TState>`** — returned by both prompt methods; use its `state` + `id` to resume later.
- **`Pipeline`**: Mapping of step names to node definitions with inputs/outputs/dependsOn

Refer to the TypeScript definitions for complete details.

## Environment Variables

- `OPENAI_API_KEY` — Your API key for OpenAI.
- `OPENAI_DEFAULT_MODEL` — Default model for the unified agent (used if no overrides are supplied).
- `OPENAI_DEFAULT_GENERATE_MODEL` — Override model for the agent/pipeline compilation phase.
