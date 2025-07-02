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
      {
        role: "system",
        content: analyzePrompt,
      },
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

    let outputText = "";

    outputText = "";
    let hasFunctionCalls = true;

    while (outputText === "" && hasFunctionCalls) {
      hasFunctionCalls = false;
      this.config.logger.log("debug", "analyze", "Prompt started", {
        prompts: input.length,
      });

      try {
        if (onMessages) {
          await onMessages(input);
        }
        const response = await (
          await this.config.openAi
        ).responses.create({
          input,
          temperature: DEFAULT_TEMPERATURE,
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
        });
        outputText = response.output_text ?? "";
        this.config.logger.log(
          "debug",
          "analyze",
          "Prompt completed",
          response.output
        );
        for (const item of response.output) {
          input.push(item);

          if (onMessages) {
            await onMessages(input);
          }
          if (item.type === "function_call" && item.call_id) {
            hasFunctionCalls = true;

            try {
              const params = JSON.parse(item.arguments);

              this.config.logger.log(
                "debug",
                "analyze",
                "Function call started",
                { name: item.name, params }
              );
              const result = await this.functionCall(page, item.name, params);

              this.config.logger.log(
                "debug",
                "analyze",
                "Function call ended",
                { name: item.name, params }
              );

              input.push({
                type: "function_call_output",
                call_id: item.call_id,
                output: JSON.stringify(result ?? {}),
              });
              if (onMessages) {
                await onMessages(input);
              }
            } catch (e: any) {
              this.config.logger.log(
                "error",
                "analyze",
                "Function call failed",
                { name: item.name, error: createErrorObject(e) }
              );
              input.push({
                type: "function_call_output",
                call_id: item.call_id,
                output: `Error while executing function: ${e.message}`,
              });

              if (onMessages) {
                await onMessages(input);
              }
            }
          }
        }
      } catch (e: any) {
        this.config.logger.log("error", "analyze", "Prompt failed", {
          error: createErrorObject(e),
        });

        this.config.logger.log("error", "analyze", "Analysis failed", {
          prompt: userPrompt,
        });

        if (onMessages) {
          await onMessages(input);
        }
        return {
          messages: input,
        };
      }
    }
    this.config.logger.log("info", "analyze", "Analysis ended", {
      analysis: outputText,
      prompt: userPrompt,
    });

    if (onMessages) {
      await onMessages(input);
    }
    return {
      result: { analysis: JSON.parse(outputText), prompt: userPrompt },
      messages: input,
    };
  }
  public async generate(
    page: Page,
    { analysis, prompt: userPrompt }: AnalysisResult,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (message: unknown[]) => Promise<void> | void
  ): Promise<AiResult<Pipeline>> {
    const ai = await this.config.openAi;

    this.config.logger.log("info", "generate", "Generation started", {
      prompt: userPrompt,
      analysis,
    });

    const input: OpenAI.Responses.ResponseInput = [
      {
        role: "system",
        content: pipelinePrompt(analysis),
      },
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
            node: {
              type: "string",
              description: "The pipeline node type, e.g., 'page::goto'",
            },
          },
          additionalProperties: false,
          required: ["node"],
        },
        strict: true,
      },
    ];

    let outputText = "";

    const performPrompt = async () => {
      this.config.logger.log("debug", "generate", "Prompt started", {
        prompts: input.length,
      });

      try {
        outputText = "";

        if (onMessages) {
          await onMessages(input);
        }

        const response = await ai.responses.create({
          input,
          temperature: DEFAULT_TEMPERATURE,
          model: this.config.models?.generate ?? PIPELINE_GENERATE_MODEL,
          text: pipelineOutput,
          tools,
          tool_choice: "auto",
        });

        outputText = response.output_text ?? "";

        this.config.logger.log(
          "debug",
          "generate",
          "Prompt completed",
          response.output
        );

        for (const item of response.output) {
          input.push(item);

          if (item.type === "function_call" && item.call_id) {
            try {
              const params = JSON.parse(item.arguments);
              this.config.logger.log(
                "debug",
                "generate",
                "Function call started",
                {
                  name: item.name,
                  params,
                }
              );

              if (item.name !== "getNodeSchema") {
                throw new TypeError(`Invalid function name: ${item.name}`);
              }

              const nodeName = params.node as `${string}::${string}`;
              const nodeSchema = pipelineSchema.additionalProperties.anyOf.find(
                (item) => item.properties.node.const === nodeName
              );

              if (!nodeSchema) {
                throw new TypeError(`Node schema was not found`);
              }

              input.push({
                type: "function_call_output",
                call_id: item.call_id,
                output: JSON.stringify(nodeSchema),
              });
              if (onMessages) await onMessages(input);
            } catch (e: any) {
              this.config.logger.log(
                "error",
                "generate",
                "Function call failed",
                {
                  name: item.name,
                  error: createErrorObject(e),
                }
              );

              input.push({
                type: "function_call_output",
                call_id: item.call_id,
                output: `Error while executing function: ${e.message}`,
              });
            }
          }
        }

        if (onMessages) await onMessages(input);
      } catch (e: any) {
        this.config.logger.log("error", "generate", "Prompt failed", {
          error: createErrorObject(e),
        });
      }
    };

    await performPrompt();

    let retryCount = 0;

    do {
      if (retryCount >= MAX_RESPONSE_FIX_RETRY) {
        const e = new Error(
          `Exceeded maximum retries (${MAX_RESPONSE_FIX_RETRY}) while validating pipeline response.`
        );
        this.config.logger.log("error", "generate", "Max retry reached", {
          error: createErrorObject(e),
        });
        return { messages: input };
      }

      let responseObject: unknown;

      try {
        responseObject = JSON.parse(outputText);
      } catch {
        this.config.logger.log(
          "error",
          "generate",
          "Failed to parse the JSON response",
          { text: outputText }
        );

        input.push({
          type: "message",
          role: "system",
          content: JSON.stringify({
            type: "parse-error",
            message: "Failed to parse the JSON response",
          }),
        });

        retryCount++;
        await performPrompt();
        continue;
      }

      if (!validatePipelineSchema(responseObject)) {
        this.config.logger.log(
          "error",
          "generate",
          "Schema validation failed",
          {
            errors: getPipelineValidationErrors(),
          }
        );

        input.push({
          type: "message",
          role: "system",
          content: JSON.stringify({
            type: "validate-error",
            errors: getPipelineValidationErrors(),
          }),
        });

        retryCount++;
        await performPrompt();
        continue;
      }

      const compileResult =
        this.config.pipelineProvider.compile(responseObject);
      if (!hasPipeline(compileResult)) {
        this.config.logger.log("error", "generate", "Compilation failed", {
          errors: compileResult.errors,
        });

        input.push({
          type: "message",
          role: "system",
          content: JSON.stringify({
            type: "pipeline-compile-errors",
            errors: compileResult.errors,
          }),
        });

        retryCount++;
        await performPrompt();
        continue;
      }

      this.config.logger.log("info", "generate", "Generation successful", {
        pipeline: compileResult.pipeline,
      });

      return {
        result: compileResult.pipeline,
        messages: input,
      };
    } while (true);
  }
}
