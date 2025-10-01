import {
  AiModel,
  AiModelMessage,
  AiResult,
  TokenUsage,
  UsageEventEnd,
  UsageEventStart,
} from "@silyze/browsary-ai-provider";
import type { OpenAiConfig } from "./index";
import { Logger } from "@silyze/logger";
import OpenAI from "openai";
import { DEFAULT_TEMPERATURE } from "./defaults";

type UsageEmitter = {
  emitStart(
    base: Omit<UsageEventStart, "phase" | "startedAt">
  ): UsageEventStart;
  emitEnd(
    base: Omit<UsageEventEnd, "phase" | "endedAt">,
    started?: number
  ): UsageEventEnd;
};

export class OpenAiModel<TModelContext> extends AiModel<TModelContext> {
  #config: OpenAiConfig;
  #model: string;
  #context: TModelContext;
  #logger: Logger;
  #usageEmitter: UsageEmitter;

  constructor(
    logger: Logger,
    model: string,
    config: OpenAiConfig,
    context: TModelContext,
    usageEmitter: UsageEmitter
  ) {
    super();
    this.#logger = logger;
    this.#config = config;
    this.#model = model;
    this.#context = context;
    this.#usageEmitter = usageEmitter;
  }

  #createInput(context: TModelContext, messages: AiModelMessage[]) {
    const promptContext = { ...this.#context, ...context };

    const input: OpenAI.Responses.ResponseInput = [
      {
        role: "system",
        content: `Prompt context:\n` + JSON.stringify(promptContext),
      },
      ...messages.map((item) => ({ role: item.type, content: item.content })),
    ];

    return input;
  }

  #mapUsage(
    usage?: OpenAI.Responses.ResponseUsage | null
  ): TokenUsage | undefined {
    if (!usage) {
      return undefined;
    }

    return {
      inputTokens: usage.input_tokens,
      outputTokens: usage.output_tokens,
      totalTokens: usage.total_tokens,
    };
  }

  async prompt(
    context: TModelContext,
    messages: AiModelMessage[]
  ): Promise<AiResult<string>> {
    const input = this.#createInput(context, messages);
    this.#logger.log("debug", "prompt", "Prompt started", {
      messages,
      context,
    });

    const start = this.#usageEmitter.emitStart({
      source: "model.prompt",
      model: this.#model,
      metadata: {
        messageCount: input.length,
      },
    });

    try {
      const response = await (
        await this.#config.openAi
      ).responses.create({
        input,
        temperature: DEFAULT_TEMPERATURE,
        model: this.#model,
      });

      const outputText = response.output_text ?? "";
      this.#logger.log("debug", "prompt", "Prompt completed", response.output);

      for (const item of response.output) {
        input.push(item);
      }

      this.#usageEmitter.emitEnd({
        source: "model.prompt",
        model: this.#model,
        startedAt: start.startedAt,
        usage: this.#mapUsage(response.usage),
        metadata: {
          messageCount: input.length,
        },
      });

      return {
        messages: input,
        result: outputText,
      };
    } catch (error) {
      this.#usageEmitter.emitEnd({
        source: "model.prompt",
        model: this.#model,
        startedAt: start.startedAt,
        metadata: {
          messageCount: input.length,
        },
      });
      throw error;
    }
  }

  async promptWithSchema<T>(
    context: TModelContext,
    messages: AiModelMessage[],
    schema: object
  ): Promise<AiResult<T>> {
    const input = this.#createInput(context, messages);
    this.#logger.log("debug", "promptWithSchema", "Prompt started", {
      messages,
      context,
    });

    const start = this.#usageEmitter.emitStart({
      source: "model.promptWithSchema",
      model: this.#model,
      metadata: {
        messageCount: input.length,
      },
    });

    try {
      const response = await (
        await this.#config.openAi
      ).responses.create({
        input,
        temperature: DEFAULT_TEMPERATURE,
        model: this.#model,
        text: {
          format: {
            type: "json_schema",
            name: "output",
            schema: schema as Record<string, unknown>,
            strict: true,
          },
        },
      });

      const outputText = response.output_text ?? "";
      this.#logger.log(
        "debug",
        "promptWithSchema",
        "Prompt completed",
        response.output
      );

      for (const item of response.output) {
        input.push(item);
      }

      this.#usageEmitter.emitEnd({
        source: "model.promptWithSchema",
        model: this.#model,
        startedAt: start.startedAt,
        usage: this.#mapUsage(response.usage),
        metadata: {
          messageCount: input.length,
        },
      });

      return {
        messages: input,
        result: JSON.parse(outputText),
      };
    } catch (error) {
      this.#usageEmitter.emitEnd({
        source: "model.promptWithSchema",
        model: this.#model,
        startedAt: start.startedAt,
        metadata: {
          messageCount: input.length,
        },
      });
      throw error;
    }
  }
}
