import OpenAI from "openai";
import { DEFAULT_TEMPERATURE, MAX_RESPONSE_FIX_RETRY } from "./defaults";
import { assert } from "@mojsoski/assert";
import {
  AiModel,
  AiProvider,
  AiResult,
  AnalysisResult,
} from "@silyze/browsary-ai-provider";
import analyzePrompt, { analyzeOutputSchema } from "./prompts/analyze";
import { ANALYZE_MODEL, analyzeTools } from "./prompts/analyze";
import {
  pipelineOutput,
  validatePipelineSchema,
  PIPELINE_GENERATE_MODEL,
  getPipelineValidationErrors,
} from "./prompts/pipeline";
import pipelinePrompt from "./prompts/pipeline";
import { Page } from "puppeteer-core";
import { type Logger } from "@silyze/logger";
import {
  GenericNode,
  hasPipeline,
  Pipeline,
  PipelineProvider,
  pipelineSchema,
} from "@silyze/browsary-pipeline";
import { OpenAiModel } from "./model";

import {
  Conversation,
  ModelConfiguration,
  FunctionConfiguration,
} from "./conversation";

export type OpenAiConfig = {
  openAi: OpenAI;
  pipelineProvider: PipelineProvider;
  logger: Logger;
  models?: {
    analyze?: string;
    generate?: string;
  };
};

export class OpenAiProvider extends AiProvider<Page, OpenAiConfig> {
  createModel<TModelContext>(
    model: string,
    context: TModelContext
  ): AiModel<TModelContext> {
    return new OpenAiModel<TModelContext>(
      this.config.logger.createScope(model),
      model,
      this.config,
      context
    );
  }

  constructor(
    config: OpenAiConfig,
    functionCall: (ctx: Page, name: string, params: any) => Promise<unknown>
  ) {
    super(
      { ...config, logger: config.logger.createScope("openai") },
      functionCall
    );

    const analyzeModel = config.models?.analyze ?? ANALYZE_MODEL;
    const generateModel = config.models?.generate ?? PIPELINE_GENERATE_MODEL;
    assert(
      OpenAiProvider.models.analyze.includes(analyzeModel),
      "Invalid analysis model"
    );
    assert(
      OpenAiProvider.models.generate.includes(generateModel),
      "Invalid generation model"
    );
  }

  static get models() {
    return {
      generic: ["gpt-4o-mini"],
      analyze: ["gpt-4o-mini"],
      generate: ["gpt-4o-mini"],
    };
  }

  public async analyze(
    page: Page,
    userPrompt: string,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (msgs: unknown[]) => void | Promise<void>
  ): Promise<AiResult<AnalysisResult>> {
    this.config.logger.log("info", "analyze", "Analysis started", {
      prompt: userPrompt,
    });

    const promptConfig: ModelConfiguration = {
      model: this.config.models?.analyze ?? ANALYZE_MODEL,
      prompt: [
        { role: "system", content: analyzePrompt },
        {
          role: "system",
          content: `Previous pipeline version:\n ${JSON.stringify(
            previousPipeline
          )}`,
        },
        {
          role: "user",
          content: `The pipeline should perform the following action:\n ${userPrompt}`,
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "analysis",
          schema: analyzeOutputSchema,
          strict: true,
        },
      },
    };

    const functionCallRef = this.functionCall;
    const functionsConfig: FunctionConfiguration = {
      tools: analyzeTools,
      async handle(call) {
        const params = JSON.parse(call.arguments);
        const result = await functionCallRef(page, call.name, params);
        return result ?? {};
      },
    };

    const convo = new Conversation({
      model: promptConfig,
      functions: functionsConfig,
      openAi: this.config.openAi,
      onMessages,
      logger: this.config.logger.createScope(promptConfig.model),
    });

    let calls = 0;
    for await (const _ of convo.start()) {
      if (calls++ > 30) {
        convo.clearTools();
      }
    }

    const output = convo.output;
    const messages = convo.history;
    if (!output) {
      this.config.logger.log("error", "analyze", "Analysis failed", {
        prompt: userPrompt,
      });
      return { messages };
    }

    const analysis = JSON.parse(output) as AnalysisResult["analysis"];
    this.config.logger.log("info", "analyze", "Analysis ended", {
      analysis,
      prompt: userPrompt,
    });

    return { result: { analysis, prompt: userPrompt }, messages };
  }

  public async generate(
    page: Page,
    { analysis, prompt: userPrompt }: AnalysisResult,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (msgs: unknown[]) => void | Promise<void>
  ): Promise<AiResult<Pipeline>> {
    let retryCount = 0;

    const initialPrompt = pipelinePrompt(analysis);
    const promptConfig: ModelConfiguration = {
      model: this.config.models?.generate ?? PIPELINE_GENERATE_MODEL,
      prompt: [
        { role: "system", content: initialPrompt },
        {
          role: "system",
          content: `Previous pipeline version:\n ${JSON.stringify(
            previousPipeline
          )}`,
        },
        {
          role: "user",
          content: `The pipeline should perform the following action:\n ${userPrompt}`,
        },
      ],
      text: pipelineOutput,
    };

    let convo: Conversation | undefined = undefined;

    const functionsConfig: FunctionConfiguration = {
      tools: [
        {
          type: "function",
          name: "getNodeSchema",
          description:
            "Retrieve the JSON schema definition of a pipeline node type.",
          parameters: {
            type: "object",
            properties: {
              node: { type: "string", description: "e.g. 'page::goto'" },
            },
            required: ["node"],
            additionalProperties: false,
          },
          strict: true,
        },
      ],
      async handle(call, history) {
        const params = JSON.parse(call.arguments);
        const nodeName = params.node as string;
        const nodeSchema = pipelineSchema.additionalProperties.anyOf.find(
          (s) => s.properties.node.const === nodeName
        );
        if (!nodeSchema)
          throw new TypeError(`Node schema for "${nodeName}" was not found`);

        const callCount = history.filter(
          (m) => m.type === "function_call_output"
        ).length;
        if (callCount >= 10) {
          convo?.clearTools();

          history.push({
            type: "message",
            role: "system",
            content: JSON.stringify({
              type: "tool-phase-complete",
              message: "Complete.",
            }),
          });
          functionsConfig.tools = [];
        }

        return nodeSchema;
      },
    };

    convo = new Conversation({
      model: promptConfig,
      functions: functionsConfig,
      openAi: this.config.openAi,
      onMessages,
      logger: this.config.logger.createScope(promptConfig.model),
    });

    while (retryCount < MAX_RESPONSE_FIX_RETRY) {
      for await (const _ of convo.start()) {
      }

      const output = convo.output;
      if (output) {
        try {
          const parsed = JSON.parse(output);
          if (!validatePipelineSchema(parsed)) {
            const errs = getPipelineValidationErrors();
            throw new Error(`Validation errors: ${errs?.join(", ")}`);
          }
          const compiled = this.config.pipelineProvider.compile(parsed);
          if (!hasPipeline(compiled)) {
            throw new Error(
              `Compile errors: ${compiled.errors
                .map((item) => JSON.stringify(item))
                .join(", ")}`
            );
          }
          return { result: compiled.pipeline!, messages: convo.history };
        } catch (e: any) {
          convo.clearOutput();
          convo.add({
            type: "message",
            role: "system",
            content: JSON.stringify({
              type: "pipeline-error",
              message: e.message,
            }),
          });
        }
      } else {
        this.config.logger.log(
          "error",
          "generate",
          "No output from generation",
          {}
        );
      }

      retryCount++;
    }

    return { messages: convo.history };
  }
}
