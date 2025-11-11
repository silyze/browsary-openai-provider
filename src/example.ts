import "dotenv/config";
import path from "path";
import crypto from "crypto";
import readline from "readline";
import { promises as fs } from "fs";

import OpenAI from "openai";
import puppeteer, { Browser, executablePath } from "puppeteer-core";
import { PipelineAgent } from "@silyze/browsary-pipeline-agent";
import {
  PipelineCompiler,
  GenericNode,
  Pipeline,
} from "@silyze/browsary-pipeline";
import {
  createJsonLogger,
  createErrorObject,
  LogSeverity,
  LoggerContext,
} from "@silyze/logger";

import { OpenAiProvider, OpenAiConversationState } from "../lib";
import {
  AiAgentConversationState,
  AiAgentControlRequest,
} from "../lib/provider-alpha";

type ConversationPaths = {
  prompt: string;
  conversation: string;
  pipeline: string;
};

type SessionState = {
  prompt: string;
  hash: string;
  paths: ConversationPaths;
  conversation?: AiAgentConversationState<OpenAiConversationState>;
  previousPipeline: Record<string, GenericNode>;
};

type RuntimeContext = {
  agent: PipelineAgent;
  provider: OpenAiProvider;
  close(): Promise<void>;
};

type CliAction = "continue" | "pause" | "resume" | "add" | "quit";

type RunOptions = {
  controlRequests?: AiAgentControlRequest[];
  auto?: boolean;
};

const STORAGE_DIR = path.resolve(process.cwd(), ".browsary-cli");
const PROMPT_CACHE_PATH = path.join(STORAGE_DIR, "prompt.last.txt");

const logger = createJsonLogger((json) =>
  console.dir(JSON.parse(json), { depth: null })
);

function log<T>(
  severity: LogSeverity,
  area: string,
  message: string,
  object?: T,
  context?: LoggerContext
) {
  logger.log(severity, area, message, object, context);
}

async function ensureStorageDir() {
  await fs.mkdir(STORAGE_DIR, { recursive: true });
}

function sha256(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

async function readJsonFile<T>(filePath: string): Promise<T | undefined> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch (error: any) {
    if (error?.code === "ENOENT") {
      return undefined;
    }
    log("warn", "fs", `Failed to read ${filePath}`, createErrorObject(error));
    return undefined;
  }
}

async function readTextFile(filePath: string): Promise<string | undefined> {
  try {
    return await fs.readFile(filePath, "utf8");
  } catch (error: any) {
    if (error?.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

function getPaths(hash: string): ConversationPaths {
  return {
    prompt: path.join(STORAGE_DIR, `prompt.${hash}.txt`),
    conversation: path.join(STORAGE_DIR, `conversation.${hash}.json`),
    pipeline: path.join(STORAGE_DIR, `pipeline.${hash}.json`),
  };
}

function meaningfulWordCount(text: string): number {
  return text
    .split(/\s+/)
    .map((word) => word.replace(/[^a-z0-9]/gi, "").trim())
    .filter((word) => word.length >= 4).length;
}

function hasUrlHint(text: string): boolean {
  return /https?:\/\/|www\.|\.[a-z]{2,3}\b/i.test(text);
}

const ACTION_VERBS = [
  "click",
  "search",
  "extract",
  "scrape",
  "fill",
  "submit",
  "download",
  "upload",
  "navigate",
  "summarize",
  "compare",
  "classify",
  "collect",
];

const OUTPUT_HINTS = [
  "report",
  "summary",
  "list",
  "table",
  "pipeline",
  "result",
  "json",
  "csv",
  "screenshot",
  "export",
];

function includesKeyword(text: string, keywords: string[]): boolean {
  return keywords.some((keyword) =>
    new RegExp(`\\b${keyword}\\b`, "i").test(text)
  );
}

function hasActionVerb(text: string): boolean {
  return includesKeyword(text, ACTION_VERBS);
}

function hasOutputHint(text: string): boolean {
  return includesKeyword(text, OUTPUT_HINTS);
}

function ask(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) =>
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.trim());
    })
  );
}

async function ensurePromptDetails(prompt: string): Promise<string> {
  const additions: string[] = [];
  const trimmed = prompt.trim();
  if (meaningfulWordCount(trimmed) < 5) {
    const goal = await ask(
      "Your request is brief. What is the end goal for this run? "
    );
    if (goal) {
      additions.push(`Goal clarification: ${goal}`);
    }
  }

  if (!hasUrlHint(trimmed)) {
    const site = await ask(
      "Which website or URL should the agent focus on? (leave blank to skip) "
    );
    if (site) {
      additions.push(`Target website: ${site}`);
    }
  }

  if (!hasActionVerb(trimmed)) {
    const actions = await ask(
      "List one or two critical actions (click, fill, export, etc.): "
    );
    if (actions) {
      additions.push(`Critical actions: ${actions}`);
    }
  }

  if (!hasOutputHint(trimmed)) {
    const output = await ask(
      "What output should the agent produce? (summary, CSV, JSON, etc.): "
    );
    if (output) {
      additions.push(`Desired output: ${output}`);
    }
  }

  if (!additions.length) {
    return trimmed;
  }

  console.log("Captured additional context for the agent:");
  additions.forEach((line, index) =>
    console.log(`  ${index + 1}. ${line}`)
  );

  return `${trimmed}\n\nAdditional details provided via CLI:\n${additions
    .map((line, index) => `${index + 1}. ${line}`)
    .join("\n")}`;
}

async function preparePrompt(
  cliPrompt?: string
): Promise<{ prompt: string; hash: string; paths: ConversationPaths }> {
  await ensureStorageDir();
  let prompt = cliPrompt?.trim() ?? "";
  const cached = (await readTextFile(PROMPT_CACHE_PATH))?.trim();

  while (!prompt) {
    const hint = cached
      ? "Enter prompt (leave blank to reuse last prompt): "
      : "Enter prompt: ";
    const answer = await ask(hint);
    if (!answer && cached) {
      prompt = cached;
    } else if (answer) {
      prompt = answer;
    }
    if (!prompt) {
      console.log("A prompt is required to continue.");
    }
  }

  const enriched = await ensurePromptDetails(prompt);
  const hash = sha256(enriched);
  const paths = getPaths(hash);

  await fs.writeFile(PROMPT_CACHE_PATH, enriched, "utf8");
  await fs.writeFile(paths.prompt, enriched, "utf8");

  return { prompt: enriched, hash, paths };
}

async function loadSession(
  prompt: string,
  hash: string,
  paths: ConversationPaths
): Promise<SessionState> {
  const conversation =
    await readJsonFile<AiAgentConversationState<OpenAiConversationState>>(
      paths.conversation
    );
  const previousPipeline =
    (await readJsonFile<Record<string, GenericNode>>(paths.pipeline)) ?? {};

  return {
    prompt,
    hash,
    paths,
    conversation: conversation ?? undefined,
    previousPipeline,
  };
}

function parseArgs(): { prompt?: string } {
  const args = process.argv.slice(2);
  const options: { prompt?: string } = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--prompt" && args[i + 1]) {
      options.prompt = args[i + 1];
      i++;
    }
  }
  return options;
}

async function createRuntime(): Promise<RuntimeContext> {
  const apiKey = process.env.OPENAI_SECRET_KEY ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("Set OPENAI_SECRET_KEY or OPENAI_API_KEY to run the CLI.");
  }

  const openAi = new OpenAI({ apiKey });
  const browserPromise = puppeteer.launch({
    headless: false,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || executablePath(),
  });

  let resolvedBrowser: Browser | undefined;
  browserPromise
    .then((browser) => {
      resolvedBrowser = browser;
      return browser;
    })
    .catch(() => {});

  const agent = new PipelineAgent({
    browser: browserPromise,
    viewport: { width: 1200, height: 800 },
  });

  const providerConfig = {
    openAi,
    pipelineProvider: new PipelineCompiler(),
    logger,
    models: {
      agent:
        process.env.AGENT_MODEL ??
        process.env.GENERATE_MODEL ??
        process.env.ANALYZE_MODEL ??
        "gpt-4o-mini",
    },
  };

  const context = (agent as any).createContext(OpenAiProvider, providerConfig);
  const provider = context.provider as OpenAiProvider;

  return {
    agent,
    provider,
    async close() {
      try {
        const browser = resolvedBrowser ?? (await browserPromise.catch(() => undefined));
        await browser?.close();
      } catch (error) {
        log("warn", "browser", "Failed to close browser", createErrorObject(error));
      }
    },
  };
}

async function persistConversation(session: SessionState) {
  if (!session.conversation) {
    return;
  }
  await fs.writeFile(
    session.paths.conversation,
    JSON.stringify(session.conversation, null, 2),
    "utf8"
  );
}

function createCallbacks(session: SessionState) {
  return {
    onPipelineUpdate: async (pipeline: Pipeline) => {
      const json = pipeline.toJSON();
      session.previousPipeline = json;
      await fs.writeFile(
        session.paths.pipeline,
        JSON.stringify(json, null, 2),
        "utf8"
      );
      const relativePath = path.relative(process.cwd(), session.paths.pipeline);
      console.log(
        `[pipeline] Saved ${Object.keys(json).length} nodes to ${relativePath}`
      );
    },
    onStatusUpdate: async (status: string) => {
      console.log(`[status] ${status}`);
    },
    onMessages: async (messages: unknown[]) => {
      log("debug", "messages", "Conversation batch received", {
        count: messages.length,
      });
    },
  };
}

async function runConversationStep(
  runtime: RuntimeContext,
  session: SessionState,
  options?: RunOptions
) {
  let requests = options?.controlRequests;
  const auto =
    options?.auto !== undefined
      ? options.auto
      : !requests || requests.length === 0;

  do {
    const callbacks = createCallbacks(session);
    const result = await runtime.agent.evaluate((ctx) =>
      session.conversation
        ? runtime.provider.continuePrompt(ctx, {
            conversation: session.conversation!,
            previousPipeline: session.previousPipeline,
            controlRequests: requests,
            ...callbacks,
          })
        : runtime.provider.prompt(ctx, {
            userPrompt: session.prompt,
            previousPipeline: session.previousPipeline,
            controlRequests: requests,
            ...callbacks,
          })
    );

    session.conversation = result;
    await persistConversation(session);
    log("info", "conversation", "Conversation state persisted", {
      status: result.status,
      isComplete: result.isComplete,
      isPaused: result.isPaused,
    });

    requests = undefined;
  } while (
    auto &&
    session.conversation &&
    !session.conversation.isComplete &&
    !session.conversation.isPaused
  );
}

type MenuOption = {
  key: string;
  label: string;
  action: CliAction;
};

function buildMenuOptions(session: SessionState): MenuOption[] {
  const options: MenuOption[] = [];
  if (session.conversation?.isPaused) {
    options.push({ key: "r", label: "Resume conversation", action: "resume" });
  } else {
    options.push({
      key: "c",
      label: session.conversation ? "Continue conversation" : "Start conversation",
      action: "continue",
    });
  }

  if (session.conversation && !session.conversation.isPaused) {
    options.push({ key: "p", label: "Pause after current step", action: "pause" });
  }

  if (session.conversation) {
    options.push({
      key: "a",
      label: "Add more instructions",
      action: "add",
    });
  }

  options.push({ key: "q", label: "Quit", action: "quit" });
  return options;
}

async function chooseAction(session: SessionState): Promise<CliAction> {
  const options = buildMenuOptions(session);
  const defaultKey = options[0].key;
  const menu = options.map((opt) => `[${opt.key}] ${opt.label}`).join("  ");
  const answer = await ask(
    `${menu}\nSelect action (default: ${defaultKey}): `
  );
  const key = (answer || defaultKey).toLowerCase();
  const choice = options.find((opt) => opt.key === key);
  if (!choice) {
    console.log("Unknown option. Please try again.");
    return chooseAction(session);
  }
  return choice.action;
}

async function handlePause(
  runtime: RuntimeContext,
  session: SessionState
): Promise<void> {
  if (!session.conversation) {
    console.log("Start a conversation before pausing.");
    return;
  }
  const reason = await ask("Pause reason (optional): ");
  await runConversationStep(runtime, session, {
    controlRequests: [{ type: "pause", reason: reason || undefined }],
    auto: false,
  });
  console.log("Conversation paused.");
}

async function handleResume(
  runtime: RuntimeContext,
  session: SessionState
): Promise<void> {
  if (!session.conversation) {
    console.log("Nothing to resume yet.");
    return;
  }
  await runConversationStep(runtime, session, {
    controlRequests: [{ type: "resume" }],
    auto: true,
  });
}

async function handleAddInstructions(
  runtime: RuntimeContext,
  session: SessionState
): Promise<void> {
  if (!session.conversation) {
    console.log("Start the conversation before adding instructions.");
    return;
  }
  const addition = await ask("Describe the extra instruction: ");
  if (!addition) {
    console.log("No instruction added.");
    return;
  }
  await runConversationStep(runtime, session, {
    controlRequests: [{ type: "addMessages", messages: [addition] }],
    auto: true,
  });
}

async function main() {
  try {
    const args = parseArgs();
    const { prompt, hash, paths } = await preparePrompt(args.prompt);
    const session = await loadSession(prompt, hash, paths);
    console.log(`Prompt hash: ${hash.slice(0, 8)}…`);

    if (session.conversation) {
      console.log(
        `Loaded previous conversation (${session.conversation.status ?? "unknown status"})`
      );
    }

    const runtime = await createRuntime();

    const shutdown = async () => {
      await runtime.close();
    };

    let running = true;
    try {
      while (running) {
        if (session.conversation?.isComplete) {
          console.log("Conversation marked complete. Pipeline persisted on disk.");
          break;
        }

        const action = await chooseAction(session);
        switch (action) {
          case "continue":
            await runConversationStep(runtime, session, { auto: true });
            break;
          case "pause":
            await handlePause(runtime, session);
            break;
          case "resume":
            await handleResume(runtime, session);
            break;
          case "add":
            await handleAddInstructions(runtime, session);
            break;
          case "quit":
            running = false;
            break;
        }
      }
    } finally {
      await shutdown();
    }
  } catch (error) {
    log("error", "cli", "Fatal error inside CLI", createErrorObject(error));
    process.exitCode = 1;
  }
}

main().catch((error) => {
  log("error", "cli", "Unhandled rejection", createErrorObject(error));
  process.exit(1);
});
