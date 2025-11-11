import {
  AiModel,
  AiModelMessage,
  AiResult,
  TokenUsage,
  UsageEventEnd,
  UsageEventStart,
} from "./provider-alpha";
import type { OpenAiConfig } from "./index";
import { Logger } from "@silyze/logger";
import OpenAI from "openai";
import { DEFAULT_TEMPERATURE } from "./defaults";

type UsageEmitter = {
  emitStartChecked(
    base: Omit<UsageEventStart, "phase" | "startedAt">
  ): Promise<{ event: UsageEventStart; proceed: boolean }>;
  emitEndChecked(
    base: Omit<UsageEventEnd, "phase" | "endedAt">,
    started?: number
  ): Promise<{ event: UsageEventEnd; proceed: boolean }>;
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
    if (!usage) return undefined;
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

    const start = await this.#usageEmitter.emitStartChecked({
      source: "model.prompt",
      model: this.#model,
      metadata: { messageCount: input.length },
    });

    if (!start.proceed) {
      this.#logger.log("debug", "prompt", "Monitor vetoed start", {
        model: this.#model,
      });
      return { messages: input, result: undefined };
    }

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

      for (const item of response.output) input.push(item);

      const end = await this.#usageEmitter.emitEndChecked(
        {
          source: "model.prompt",
          model: this.#model,
          startedAt: start.event.startedAt,
          usage: this.#mapUsage(response.usage),
          metadata: { messageCount: input.length },
        },
        start.event.startedAt
      );

      if (!end.proceed) {
        this.#logger.log("debug", "prompt", "Monitor vetoed end", {
          model: this.#model,
        });
        return { messages: input, result: undefined };
      }

      return { messages: input, result: outputText };
    } catch (error) {
      await this.#usageEmitter.emitEndChecked(
        {
          source: "model.prompt",
          model: this.#model,
          startedAt: start.event.startedAt,
          metadata: { messageCount: input.length },
        },
        start.event.startedAt
      );
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

    const start = await this.#usageEmitter.emitStartChecked({
      source: "model.promptWithSchema",
      model: this.#model,
      metadata: { messageCount: input.length },
    });

    if (!start.proceed) {
      this.#logger.log("debug", "promptWithSchema", "Monitor vetoed start", {
        model: this.#model,
      });
      return { messages: input, result: undefined as unknown as T };
    }

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

      for (const item of response.output) input.push(item);

      const end = await this.#usageEmitter.emitEndChecked(
        {
          source: "model.promptWithSchema",
          model: this.#model,
          startedAt: start.event.startedAt,
          usage: this.#mapUsage(response.usage),
          metadata: { messageCount: input.length },
        },
        start.event.startedAt
      );

      if (!end.proceed) {
        this.#logger.log("debug", "promptWithSchema", "Monitor vetoed end", {
          model: this.#model,
        });
        return { messages: input, result: undefined as unknown as T };
      }

      return { messages: input, result: JSON.parse(outputText) as T };
    } catch (error) {
      await this.#usageEmitter.emitEndChecked(
        {
          source: "model.promptWithSchema",
          model: this.#model,
          startedAt: start.event.startedAt,
          metadata: { messageCount: input.length },
        },
        start.event.startedAt
      );
      throw error;
    }
  }
}
