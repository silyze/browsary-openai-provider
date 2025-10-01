import OpenAI from "openai";
import { DEFAULT_TEMPERATURE } from "./defaults";
import { assert } from "@mojsoski/assert";
import { createErrorObject, type Logger } from "@silyze/logger";
import { TokenUsage } from "@silyze/browsary-ai-provider";

export interface FunctionConfiguration {
  tools: OpenAI.Responses.Tool[];
  handle(
    call: OpenAI.Responses.ResponseFunctionToolCall,
    history: OpenAI.Responses.ResponseInput
  ): Promise<unknown>;
}

export interface ModelConfiguration {
  model: string;
  temperature?: number;
  prompt: OpenAI.Responses.ResponseInput;
  text: OpenAI.Responses.ResponseTextConfig;
}

export interface ConversationConfiguration {
  functions: FunctionConfiguration;
  model: ModelConfiguration;
  openAi: OpenAI;
  logger: Logger;
  onMessages?: (messages: unknown[]) => void | Promise<void>;
}

export class Conversation {
  #config: ConversationConfiguration;
  #history: OpenAI.Responses.ResponseInput;
  #output: string | undefined;
  #usage: TokenUsage | undefined;

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

  #pushMessage(
    message:
      | OpenAI.Responses.ResponseOutputItem
      | OpenAI.Responses.ResponseInputMessageItem
      | OpenAI.Responses.ResponseInputItem
      | OpenAI.Responses.ResponseItem
  ) {
    this.#history.push(message);
    if (this.#config.onMessages) {
      this.#config.onMessages(this.#history);
    }
  }

  get history() {
    return this.#history;
  }

  get output() {
    return this.#output;
  }

  get usage() {
    return this.#usage;
  }

  constructor(config: ConversationConfiguration) {
    this.#config = config;
    this.#history = config.model.prompt;
    this.#config.logger.log(
      "info",
      "conversation",
      "Initialized conversation",
      { model: this.#config.model.model, promptCount: this.#history.length }
    );
  }

  add(message: OpenAI.Responses.ResponseInputItem) {
    this.#config.logger.log(
      "debug",
      "conversation",
      "Adding user/system message",
      { message }
    );
    this.#pushMessage(message);
  }

  clearTools() {
    this.#config.logger.log("debug", "conversation", "Clearing tools", {
      previousTools: this.#config.functions.tools,
    });
    this.#config.functions.tools = [];
  }

  clearOutput() {
    this.#config.logger.log(
      "debug",
      "conversation",
      "Clearing previous output"
    );
    this.#output = undefined;
  }

  async *start() {
    assert(
      this.#output === undefined,
      "Cannot start a conversation that is already complete"
    );

    let running = true;
    const { model, temperature } = this.#config.model;

    while (running) {
      this.#config.logger.log(
        "debug",
        "conversation",
        "Sending request to OpenAI",
        {
          model,
          promptCount: this.#history.length,
          toolCount: this.#config.functions.tools.length,
        }
      );

      yield;
      const response = await this.#config.openAi.responses.create({
        model,
        input: this.#history,
        temperature: temperature ?? DEFAULT_TEMPERATURE,
        tools: this.#config.functions.tools,
        text: this.#config.model.text,
      });

      this.#usage = this.#mapUsage(response.usage);

      this.#config.logger.log(
        "debug",
        "conversation",
        "Received response from OpenAI",
        {
          outputItems: response.output.length,
          outputText: response.output_text,
        }
      );

      for (const message of response.output) {
        this.#pushMessage(message);

        if (message.type === "function_call") {
          this.#config.logger.log(
            "debug",
            "conversation",
            "Handling function call",
            { name: message.name, args: message.arguments }
          );

          try {
            const result = await this.#config.functions.handle(
              message,
              this.#history
            );
            this.#config.logger.log(
              "debug",
              "conversation",
              "Function call succeeded",
              { name: message.name }
            );
            this.#pushMessage({
              id: undefined!,
              type: "function_call_output",
              call_id: message.call_id,
              output: JSON.stringify(result),
            });
          } catch (e: any) {
            this.#config.logger.log(
              "error",
              "conversation",
              "Function call failed",
              { name: message.name, error: createErrorObject(e) }
            );
            this.#pushMessage({
              id: undefined!,
              type: "function_call_output",
              call_id: message.call_id,
              output: `Error while executing function: ${e.message}`,
            });
          }
          continue;
        }

        if (message.type === "message") {
          running = false;
          this.#output = response.output_text;
          this.#config.logger.log(
            "info",
            "conversation",
            "Conversation complete",
            { output: this.#output }
          );
          break;
        }
      }
    }
  }
}
