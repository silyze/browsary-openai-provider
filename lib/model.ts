import {
  AiModel,
  AiModelMessage,
  AiResult,
} from "@silyze/browsary-ai-provider";
import type { OpenAiConfig } from "./index";
import { Logger } from "@silyze/logger";
import OpenAI from "openai";
import { DEFAULT_TEMPERATURE } from "./defaults";

export class OpenAiModel<TModelContext> extends AiModel<TModelContext> {
  #config: OpenAiConfig;
  #model: string;
  #context: TModelContext;
  #logger: Logger;

  constructor(
    logger: Logger,
    model: string,
    config: OpenAiConfig,
    context: TModelContext
  ) {
    super();
    this.#logger = logger;
    this.#config = config;
    this.#model = model;
    this.#context = context;
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

  async prompt(
    context: TModelContext,
    messages: AiModelMessage[]
  ): Promise<AiResult<string>> {
    const input = this.#createInput(context, messages);
    this.#logger.log("info", "analyze", "Prompt started", {
      messages,
      context,
    });

    const response = await (
      await this.#config.openAi
    ).responses.create({
      input,
      temperature: DEFAULT_TEMPERATURE,
      model: this.#model,
    });

    const outputText = response.output_text ?? "";
    this.#logger.log("debug", "analyze", "Prompt completed", response.output);

    for (const item of response.output) {
      input.push(item);
    }

    return {
      messages: input,
      result: outputText,
    };
  }

  async promptWithSchema<T>(
    context: TModelContext,
    messages: AiModelMessage[],
    schema: object
  ): Promise<AiResult<T>> {
    const input = this.#createInput(context, messages);
    this.#logger.log("info", "analyze", "Prompt started", {
      messages,
      context,
    });

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
    this.#logger.log("debug", "analyze", "Prompt completed", response.output);

    for (const item of response.output) {
      input.push(item);
    }

    return {
      messages: input,
      result: JSON.parse(outputText),
    };
  }
}
