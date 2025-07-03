import {
  AiModel,
  AiProvider,
  AiResult,
  AnalysisResult,
} from "@silyze/browsary-ai-provider";
import OpenAI from "openai";
import analyzePrompt, { analyzeOutputSchema } from "./prompts/analyze";
import { ANALYZE_MODEL, analyzeTools } from "./prompts/analyze";
import { Page } from "puppeteer-core";
import pipelinePrompt from "./prompts/pipeline";
import {
  pipelineOutput,
  validatePipelineSchema,
  PIPELINE_GENERATE_MODEL,
  getPipelineValidationErrors,
} from "./prompts/pipeline";
import { createErrorObject, type Logger } from "@silyze/logger";
import {
  GenericNode,
  hasPipeline,
  Pipeline,
  PipelineCompileResult,
  PipelineProvider,
  pipelineSchema,
} from "@silyze/browsary-pipeline";
import { assert } from "@mojsoski/assert";
import { OpenAiModel } from "./model";
import { DEFAULT_TEMPERATURE, MAX_RESPONSE_FIX_RETRY } from "./defaults";
import { FunctionTool } from "../node_modules/openai/src/resources/responses/responses";

export type OpenAiConfig = {
  openAi: OpenAI | Promise<OpenAI>;
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
  public constructor(
    config: OpenAiConfig,
    functionCall: (context: Page, name: string, params: any) => Promise<unknown>
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

  #safeJsonParse(text: string): unknown | undefined {
    try {
      return JSON.parse(text);
    } catch {
      return undefined;
    }
  }

  #addSystemMessage(input: OpenAI.Responses.ResponseInput, payload: unknown) {
    input.push({
      type: "message",
      role: "system",
      content: JSON.stringify(payload),
    });
  }

  async #retryUntilValidResponse(
    page: Page,
    input: OpenAI.Responses.ResponseInput,
    config: {
      model: string;
      tools: FunctionTool[];
      text: OpenAI.Responses.ResponseTextConfig;
      onMessages?: (messages: unknown[]) => Promise<void> | void;
      handleFunctionCall: (
        item: OpenAI.Responses.ResponseFunctionToolCall,
        input: OpenAI.Responses.ResponseInput
      ) => Promise<void>;
      validate: (json: unknown) => boolean;
      compile: (json: unknown) => PipelineCompileResult;
      errorTypes: {
        parse: string;
        validate: string;
        compile: string;
      };
    }
  ): Promise<AiResult<Pipeline>> {
    let retryCount = 0;

    while (retryCount < MAX_RESPONSE_FIX_RETRY) {
      const { outputText, messages } =
        await this.#runWithPromptAndFunctionCalls(input, config);

      if (!outputText) return { messages };

      const parsed = this.#safeJsonParse(outputText);
      if (!parsed) {
        this.#addSystemMessage(input, {
          type: config.errorTypes.parse,
          message: "Failed to parse the JSON response",
        });
        retryCount++;
        continue;
      }

      if (!config.validate(parsed)) {
        this.#addSystemMessage(input, {
          type: config.errorTypes.validate,
          errors: getPipelineValidationErrors(),
        });
        retryCount++;
        continue;
      }

      const result = config.compile(parsed);
      if (!hasPipeline(result)) {
        this.#addSystemMessage(input, {
          type: config.errorTypes.compile,
          errors: result.errors,
        });
        retryCount++;
        continue;
      }

      return {
        result: result.pipeline!,
        messages,
      };
    }

    const error = new Error(
      `Exceeded maximum retries (${MAX_RESPONSE_FIX_RETRY}) while validating response.`
    );
    this.config.logger.log("error", config.model, "Max retry reached", {
      error: createErrorObject(error),
    });

    return { messages: input };
  }

  async #runWithPromptAndFunctionCalls<T>(
    input: OpenAI.Responses.ResponseInput,
    config: {
      model: string;
      tools: FunctionTool[];
      text: OpenAI.Responses.ResponseTextConfig;
      onMessages?: (messages: unknown[]) => Promise<void> | void;
      handleFunctionCall?: (
        item: OpenAI.Responses.ResponseFunctionToolCall,
        input: OpenAI.Responses.ResponseInput
      ) => Promise<void>;
    }
  ): Promise<{ outputText: string; messages: OpenAI.Responses.ResponseInput }> {
    const ai = await this.config.openAi;
    const { model, tools, text, onMessages, handleFunctionCall } = config;

    let outputText = "";
    let hasFunctionCalls = true;

    while (outputText === "" && hasFunctionCalls) {
      hasFunctionCalls = false;

      this.config.logger.log("debug", model, "Prompt started", {
        prompts: input.length,
      });

      try {
        if (onMessages) await onMessages(input);

        const response = await ai.responses.create({
          input,
          temperature: DEFAULT_TEMPERATURE,
          model,
          tools,
          text,
        });

        outputText = response.output_text ?? "";
        this.config.logger.log(
          "debug",
          model,
          "Prompt completed",
          response.output
        );

        for (const item of response.output) {
          input.push(item);
          if (onMessages) await onMessages(input);

          if (
            item.type === "function_call" &&
            item.call_id &&
            handleFunctionCall
          ) {
            hasFunctionCalls = true;
            try {
              await handleFunctionCall(item, input);
              if (onMessages) await onMessages(input);
            } catch (e: any) {
              this.config.logger.log("error", model, "Function call failed", {
                name: item.name,
                error: createErrorObject(e),
              });
              input.push({
                type: "function_call_output",
                call_id: item.call_id,
                output: `Error while executing function: ${e.message}`,
              });
              if (onMessages) await onMessages(input);
            }
          }
        }
      } catch (e: any) {
        this.config.logger.log("error", model, "Prompt failed", {
          error: createErrorObject(e),
        });
        break;
      }
    }

    return { outputText, messages: input };
  }
  public async analyze(
    page: Page,
    userPrompt: string,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (message: unknown[]) => Promise<void> | void
  ): Promise<AiResult<AnalysisResult>> {
    this.config.logger.log("info", "analyze", "Analysis started", {
      prompt: userPrompt,
    });

    const input: OpenAI.Responses.ResponseInput = [
      { role: "system", content: analyzePrompt },
      {
        role: "system",
        content: `Previous pipeline version:\n ${JSON.stringify(
          previousPipeline
        )}`,
      },
      {
        role: "user",
        content: `The pipeline should preform the following action:\n ${userPrompt}`,
      },
    ];

    const { outputText, messages } = await this.#runWithPromptAndFunctionCalls(
      input,
      {
        model: this.config.models?.analyze ?? ANALYZE_MODEL,
        tools: analyzeTools,
        text: {
          format: {
            type: "json_schema",
            name: "analysis",
            schema: analyzeOutputSchema,
            strict: true,
          },
        },
        onMessages,
        handleFunctionCall: async (item, input) => {
          const params = JSON.parse(item.arguments);
          this.config.logger.log("debug", "analyze", "Function call started", {
            name: item.name,
            params,
          });
          const result = await this.functionCall(page, item.name, params);
          this.config.logger.log("debug", "analyze", "Function call ended", {
            name: item.name,
            params,
          });
          input.push({
            type: "function_call_output",
            call_id: item.call_id,
            output: JSON.stringify(result ?? {}),
          });
        },
      }
    );

    if (!outputText) {
      this.config.logger.log("error", "analyze", "Analysis failed", {
        prompt: userPrompt,
      });
      return { messages };
    }

    this.config.logger.log("info", "analyze", "Analysis ended", {
      analysis: outputText,
      prompt: userPrompt,
    });

    return {
      result: { analysis: JSON.parse(outputText), prompt: userPrompt },
      messages,
    };
  }

  public async generate(
    page: Page,
    { analysis, prompt: userPrompt }: AnalysisResult,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (message: unknown[]) => Promise<void> | void
  ): Promise<AiResult<Pipeline>> {
    const input: OpenAI.Responses.ResponseInput = [
      { role: "system", content: pipelinePrompt(analysis) },
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
    ];

    const tools: FunctionTool[] = [
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
    ];

    const seenSchemas = new Set<string>();

    return this.#retryUntilValidResponse(page, input, {
      model: this.config.models?.generate ?? PIPELINE_GENERATE_MODEL,
      tools,
      text: pipelineOutput,
      onMessages,
      handleFunctionCall: async (item, input) => {
        const params = JSON.parse(item.arguments);
        const nodeName = params.node as `${string}::${string}`;

        if (seenSchemas.has(nodeName)) {
          this.#addSystemMessage(input, {
            type: "redundant-tool-call",
            message: `⚠️ Schema for "${nodeName}" was already retrieved. Continue with generation.`,
          });
          return;
        }
        seenSchemas.add(nodeName);

        const nodeSchema = pipelineSchema.additionalProperties.anyOf.find(
          (s) => s.properties.node.const === nodeName
        );

        if (!nodeSchema) {
          throw new TypeError(`Node schema for "${nodeName}" was not found`);
        }

        input.push({
          type: "function_call_output",
          call_id: item.call_id,
          output: JSON.stringify(nodeSchema),
        });

        if (
          input.filter((m) => m.type === "function_call_output").length >= 10 &&
          !input.some(
            (m) =>
              m.type === "message" &&
              m.role === "system" &&
              typeof m.content === "string" &&
              m.content.includes("begin generating")
          )
        ) {
          this.#addSystemMessage(input, {
            type: "tool-phase-complete",
            message:
              "✅ You have retrieved enough node schemas. Now begin generating the final pipeline JSON that fulfills the user's prompt.",
          });
        }
      },
      validate: validatePipelineSchema,
      compile: (json) => this.config.pipelineProvider.compile(json),
      errorTypes: {
        parse: "parse-error",
        validate: "validate-error",
        compile: "pipeline-compile-errors",
      },
    });
  }
}
