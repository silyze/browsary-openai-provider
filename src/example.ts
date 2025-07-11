import "dotenv/config";
import { promises as fs } from "fs";
import readline from "readline";
import crypto from "crypto";

import { PipelineAgent } from "@silyze/browsary-pipeline-agent";
import puppeteer, { executablePath } from "puppeteer-core";
import { OpenAiProvider } from "../lib";
import OpenAI from "openai";
import {
  createJsonLogger,
  createErrorObject,
  LogSeverity,
  LoggerContext,
} from "@silyze/logger";
import { PipelineCompiler } from "@silyze/browsary-pipeline";
import { assertNonNull } from "@mojsoski/assert";

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

async function getPrompt(): Promise<string> {
  const promptFile = "prompt.temp.txt";
  try {
    log("info", "prompt", "Reading prompt cache", { path: promptFile });
    const cached = await fs.readFile(promptFile, "utf8");
    log("info", "prompt", "Prompt cache hit", { prompt: cached });
    return cached.trim();
  } catch {
    log("info", "prompt", "Prompt cache miss, asking user");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    const answer = await new Promise<string>((resolve) =>
      rl.question("Enter prompt: ", (ans) => {
        rl.close();
        resolve(ans);
      })
    );
    const trimmed = answer.trim();
    await fs.writeFile(promptFile, trimmed, "utf8");
    log("info", "prompt", "Wrote prompt to cache", { path: promptFile });
    return trimmed;
  }
}

function sha256(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

function createAgent() {
  log("info", "browser", "Launching Puppeteer", {
    headless: false,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || executablePath(),
  });
  const browser = puppeteer.launch({
    headless: false,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || executablePath(),
  });
  return new PipelineAgent({
    browser,
    viewport: { height: 600, width: 800 },
  });
}

async function main() {
  try {
    const prompt = await getPrompt();
    const hash = sha256(prompt);
    log("info", "hash", "Computed prompt hash", { hash });

    const analysisPath = `analysis.${hash}.temp.json`;
    const pipelinePath = `pipeline.${hash}.temp.json`;

    try {
      log("info", "cache", "Reading pipeline cache", { path: pipelinePath });
      const rawPipeline = await fs.readFile(pipelinePath, "utf8");
      const pipelineJson = JSON.parse(rawPipeline);
      log("info", "cache", "Pipeline cache hit", { path: pipelinePath });
      return pipelineJson;
    } catch {
      log("info", "cache", "No pipeline cache found", { path: pipelinePath });
    }

    const agent = createAgent();
    const { provider } = agent.createContext(OpenAiProvider, {
      openAi: new OpenAI({ apiKey: process.env.OPENAI_SECRET_KEY }),
      pipelineProvider: new PipelineCompiler(),
      logger,
      models: {
        analyze: process.env.ANALYZE_MODEL ?? "gpt-4o-mini",
        generate: process.env.GENERATE_MODEL ?? "gpt-4o-mini",
      },
    });

    let analysisResult: unknown;
    try {
      log("info", "cache", "Reading analysis cache", { path: analysisPath });
      const rawAnalysis = await fs.readFile(analysisPath, "utf8");
      analysisResult = JSON.parse(rawAnalysis);
      log("info", "cache", "Analysis cache hit", { path: analysisPath });
    } catch {
      log("info", "cache", "No analysis cache found", { path: analysisPath });
      analysisResult = (
        await agent.evaluate((ctx) => provider.analyze(ctx, prompt, {}))
      ).result;
      await fs.writeFile(
        analysisPath,
        JSON.stringify(analysisResult, null, 2),
        "utf8"
      );
      log("info", "cache", "Wrote analysis cache", { path: analysisPath });
    }

    assertNonNull(analysisResult, "analysisResult");

    log("info", "generate", "Invoking provider.generate", { hash });
    const { result: pipeline } = await agent.evaluate((ctx) =>
      provider.generate(ctx, analysisResult as any, {})
    );
    assertNonNull(pipeline, "pipeline");

    const pipelineJson = pipeline.toJSON();
    await fs.writeFile(
      pipelinePath,
      JSON.stringify(pipelineJson, null, 2),
      "utf8"
    );
    log("info", "cache", "Wrote pipeline cache", { path: pipelinePath });
    return pipelineJson;
  } catch (e: unknown) {
    log("error", "main", "Unhandled exception", createErrorObject(e));
    throw e;
  }
}

main()
  .then((output) => {
    log("info", "main", "Finished successfully", { output });
    console.dir(output, { depth: null });
    process.exit(0);
  })
  .catch(() => {
    process.exit(1);
  });
