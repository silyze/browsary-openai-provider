import OpenAI from "openai";
import { randomUUID } from "crypto";
import { assert } from "@mojsoski/assert";
import {
  AiModel,
  AiProvider,
  AiAgentConversationState,
  PromptParams,
  ContinuePromptParams,
  AiAgentControlRequest,
  UsageMonitor,
  PipelineConversationCallbacks,
  FunctionCallStatusOptions,
} from "@silyze/browsary-ai-provider";
import { analyzeTools } from "./prompts/analyze";
import {
  validatePipelineSchema,
  PIPELINE_GENERATE_MODEL,
  getPipelineValidationErrors,
} from "./prompts/pipeline";
import agentPrompt from "./prompts/agent";
import {
  buildFunctionPromptSections,
  FunctionPromptSections,
} from "./prompts/functions";
import { Page } from "puppeteer-core";
import { type Logger, createErrorObject } from "@silyze/logger";
import {
  GenericNode,
  hasPipeline,
  Pipeline,
  PipelineProvider,
  PipelineFunctionProvider,
  PipelineFunction,
  pipelineSchema,
  RefType,
  typeDescriptor,
} from "@silyze/browsary-pipeline";
import { OpenAiModel } from "./model";

import {
  Conversation,
  ModelConfiguration,
  FunctionConfiguration,
} from "./conversation";
import {
  createPipelineToolSchema,
  getBuiltinNodeSchema,
} from "./schema-tools";

type OutputSchema = {
  anyOf: Array<Record<string, unknown>>;
};

const dependsOnSchema: Record<string, unknown> = {
  description:
    "Defines execution dependencies. Can be a string (node ID), an object reference to a boolean output, or an array of either.",
  anyOf: [
    {
      type: "string",
      description: "Node ID this node depends on unconditionally.",
    },
    {
      type: "object",
      description:
        "Conditional dependency. This node will only run if the referenced output is truthy.",
      properties: {
        nodeName: {
          type: "string",
          description: "Name of the node that produces the output.",
        },
        outputName: {
          type: "string",
          description: "Name of the boolean output to evaluate.",
        },
      },
      required: ["nodeName", "outputName"],
      additionalProperties: false,
    },
    {
      type: "array",
      description:
        "List of dependencies, each being either a node ID or a conditional output reference.",
      items: {
        anyOf: [
          {
            type: "string",
            description: "Node ID this node depends on.",
          },
          {
            type: "object",
            description:
              "Conditional dependency based on output of another node.",
            properties: {
              nodeName: {
                type: "string",
                description: "Name of the node producing the output.",
              },
              outputName: {
                type: "string",
                description: "Name of the boolean output to evaluate.",
              },
            },
            required: ["nodeName", "outputName"],
            additionalProperties: false,
          },
        ],
      },
    },
  ],
};

function descriptorToSchema(refType: string): Record<string, unknown> {
  const descriptor =
    typeDescriptor[refType as keyof typeof typeDescriptor] ?? null;

  if (Array.isArray(descriptor)) {
    return { enum: [...descriptor] };
  }

  if (descriptor && typeof descriptor === "object") {
    return { ...descriptor };
  }

  if (typeof descriptor === "string") {
    return { type: descriptor };
  }

  return {};
}

function createOutputReferenceSchema(
  refType: string,
  description?: string
): OutputSchema {
  const summary = description
    ? `${description} (type '${refType}')`
    : `Output reference of type '${refType}'.`;

  return {
    anyOf: [
      {
        type: "string",
        description: summary,
        [RefType]: refType,
      },
      {
        type: "object",
        description:
          "Dynamic output reference to input of another node (type '" +
          refType +
          "').",
        properties: {
          nodeName: { type: "string" },
          inputName: { type: "string" },
        },
        required: ["nodeName", "inputName"],
        additionalProperties: false,
        [RefType]: refType,
      },
    ],
  };
}

function createOutputOfInputSchema(refType: string): Record<string, unknown> {
  return {
    type: "object",
    properties: {
      type: {
        const: "outputOf",
        type: "string",
        [RefType]: refType,
      },
      nodeName: { type: "string" },
      outputName: { type: "string" },
    },
    required: ["type", "nodeName", "outputName"],
    additionalProperties: false,
  };
}

function createConstantInputSchema(
  valueSchema: Record<string, unknown>
): Record<string, unknown> {
  return {
    type: "object",
    properties: {
      type: { const: "constant", type: "string" },
      value: valueSchema,
    },
    required: ["type", "value"],
    additionalProperties: false,
  };
}

function buildArgsValueSchema(
  inputs: PipelineFunction["inputs"]
): Record<string, unknown> {
  const properties = Object.fromEntries(
    inputs.map((input) => {
      const schema = descriptorToSchema(input.refType);
      if (input.description) {
        return [
          input.name,
          {
            ...schema,
            description: input.description,
          },
        ] as const;
      }
      return [input.name, schema] as const;
    })
  );

  return {
    type: "object",
    description: "Function argument values keyed by input name.",
    properties,
    required: inputs.map((input) => input.name),
    additionalProperties: false,
  };
}

function buildFunctionCallSchema(
  identifier: string,
  fn: PipelineFunction
): Record<string, unknown> {
  const argsValueSchema = buildArgsValueSchema(fn.inputs);
  const identifierInputSchema = {
    anyOf: [
      createOutputOfInputSchema("string"),
      createConstantInputSchema({ type: "string", const: identifier }),
    ],
  };

  const argsInputSchema = {
    anyOf: [
      createOutputOfInputSchema("object"),
      createConstantInputSchema(argsValueSchema),
    ],
  };

  const outputEntries = new Map<string, OutputSchema>();

  for (const output of fn.outputs) {
    outputEntries.set(
      output.name,
      createOutputReferenceSchema(output.refType, output.description)
    );
  }

  const resultDescription =
    fn.outputType === "iterator"
      ? "Aggregated iterator results (type 'any')."
      : "Return value of the function (type 'any').";

  outputEntries.set(
    "result",
    createOutputReferenceSchema("any", resultDescription)
  );

  const outputs = Object.fromEntries(outputEntries);

  return {
    title: fn.metadata?.title
      ? `Call function: ${fn.metadata.title}`
      : `Call function: ${identifier}`,
    description:
      fn.metadata?.description ??
      `Invoke reusable pipeline function '${identifier}'.`,
    type: "object",
    properties: {
      node: {
        type: "string",
        const: "functions::call",
        description:
          "Unique identifier in the format 'prefix::action'. Always 'functions::call' for function invocations.",
      },
      dependsOn: dependsOnSchema,
      inputs: {
        type: "object",
        description:
          "Input bindings for this node. Keys map to inputs declared in the node type.",
        properties: {
          identifier: identifierInputSchema,
          args: argsInputSchema,
        },
        required: ["identifier", "args"],
        additionalProperties: false,
      },
      outputs: {
        type: "object",
        description:
          "Outputs produced by this node. Provide bindings for each declared output.",
        properties: outputs,
        required: Array.from(outputEntries.keys()),
        additionalProperties: false,
      },
    },
    required: ["node", "inputs", "outputs", "dependsOn"],
    additionalProperties: false,
  };
}

type ConversationPhase =
  | "idle"
  | "acting"
  | "awaiting-user"
  | "complete"
  | "paused"
  | "error";

type AgentOutputData = {
  description?: string;
  data?: unknown;
  final?: boolean;
};

type AgentChatMessage = {
  message: string;
  audience: "user" | "observers";
  at: number;
};

type AgentRunVerification = {
  success: boolean;
  errors?: unknown[];
  note?: string;
  timestamp: number;
};

type AgentArtifactState = {
  messages?: unknown[];
  lastMessage?: string;
  lastTool?: string;
  pendingQuestion?: string;
  outputData?: AgentOutputData;
  chatLog?: AgentChatMessage[];
  lastVerification?: AgentRunVerification;
};

type EmittedPipelineResult = {
  raw: Record<string, GenericNode>;
  pipeline: Pipeline;
};

type AgentRunResult = {
  messages: unknown[];
  outputText?: string;
  emittedPipeline?: EmittedPipelineResult;
  pendingQuestion?: string;
  outputData?: AgentOutputData;
  chatMessages: AgentChatMessage[];
  verification?: AgentRunVerification;
  finishRequested?: boolean;
  lastTool?: string;
};

type PersistedConversationState = {
  prompt: string;
  additionalInstructions?: string[];
  phase: ConversationPhase;
  resumePhase?: ConversationPhase;
  agent?: AgentArtifactState;
  pipeline?: {
    json?: Record<string, GenericNode>;
    messages?: unknown[];
    updatedAt?: number;
  };
  retries?: number;
  pausedReason?: string;
  metadata?: Record<string, unknown>;
};

export type OpenAiConversationState = PersistedConversationState;

type PromptWorkflowParams = {
  page: Page;
  conversationId: string;
  state: PersistedConversationState;
  previousPipeline: Record<string, GenericNode>;
  callbacks: PipelineConversationCallbacks;
  controlRequests?: AiAgentControlRequest[];
  abortController?: AbortController;
};

export type OpenAiConfig = {
  openAi: OpenAI;
  pipelineProvider: PipelineProvider;
  functionProvider?: PipelineFunctionProvider;
  logger: Logger;
  models?: {
    analyze?: string;
    generate?: string;
    agent?: string;
    status?: string;
  };
};

export class OpenAiProvider extends AiProvider<Page, OpenAiConfig> {
  private functionPromptSectionsPromise?: Promise<
    FunctionPromptSections | undefined
  >;
  private agentModel: string;
  private statusModel: string;

  private throwIfAborted(abortController?: AbortController) {
    if (!abortController?.signal.aborted) {
      return;
    }

    const reason = abortController.signal.reason;
    if (reason instanceof Error) {
      throw reason;
    }

    throw new Error(typeof reason === "string" ? reason : "Operation aborted");
  }

  private resolveFunctionProvider(): PipelineFunctionProvider | undefined {
    if (this.config.functionProvider) {
      return this.config.functionProvider;
    }

    const candidate = (
      this.config.pipelineProvider as Partial<{
        functionProvider?: PipelineFunctionProvider;
      }>
    ).functionProvider;

    return candidate;
  }

  private getFunctionPromptSections(): Promise<
    FunctionPromptSections | undefined
  > {
    if (!this.functionPromptSectionsPromise) {
      this.functionPromptSectionsPromise = (async () => {
        const provider = this.resolveFunctionProvider();
        if (!provider) {
          return undefined;
        }

        try {
          return await buildFunctionPromptSections(provider);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          this.config.logger.log(
            "warn",
            "prompt",
            "Failed to build function index",
            { message }
          );
          return undefined;
        }
      })();
    }

    return this.functionPromptSectionsPromise;
  }

  private async getFunctionNodeSchema(
    identifier: string
  ): Promise<Record<string, unknown> | undefined> {
    const provider = this.resolveFunctionProvider();
    if (!provider) {
      return undefined;
    }

    const separatorIndex = identifier.indexOf("::");
    if (separatorIndex === -1) {
      return undefined;
    }

    const namespace = identifier.slice(0, separatorIndex);
    const name = identifier.slice(separatorIndex + 2);

    if (!namespace || !name) {
      return undefined;
    }

    try {
      const fn = await provider.getFunction(namespace, name);
      if (!fn) {
        return undefined;
      }

      return buildFunctionCallSchema(identifier, fn);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.config.logger.log(
        "warn",
        "generate",
        "Failed to resolve function schema",
        { identifier, message }
      );
      return undefined;
    }
  }

  private createInitialState(prompt: string): PersistedConversationState {
    return {
      prompt,
      phase: "idle",
    };
  }

  private normalizeState(
    state: unknown,
    fallbackPrompt?: string
  ): PersistedConversationState {
    if (!state || typeof state !== "object") {
      if (!fallbackPrompt) {
        throw new Error("Conversation prompt is missing from state.");
      }
      return this.createInitialState(fallbackPrompt);
    }

    const snapshot = state as Partial<PersistedConversationState>;
    const prompt = snapshot.prompt ?? fallbackPrompt;
    if (!prompt) {
      throw new Error("Conversation prompt is required to continue.");
    }

    return {
      prompt,
      additionalInstructions: snapshot.additionalInstructions
        ? [...snapshot.additionalInstructions]
        : undefined,
      phase: snapshot.phase ?? "idle",
      resumePhase: snapshot.resumePhase,
      agent: snapshot.agent
        ? {
            ...snapshot.agent,
            messages: snapshot.agent.messages
              ? [...snapshot.agent.messages]
              : undefined,
            chatLog: snapshot.agent.chatLog
              ? snapshot.agent.chatLog.map((entry) => ({ ...entry }))
              : undefined,
            outputData: snapshot.agent.outputData
              ? { ...snapshot.agent.outputData }
              : undefined,
            lastVerification: snapshot.agent.lastVerification
              ? { ...snapshot.agent.lastVerification }
              : undefined,
          }
        : undefined,
      pipeline: snapshot.pipeline ? { ...snapshot.pipeline } : undefined,
      retries: snapshot.retries,
      pausedReason: snapshot.pausedReason,
      metadata: snapshot.metadata ? { ...snapshot.metadata } : undefined,
    };
  }

  private cloneState(
    state: PersistedConversationState
  ): PersistedConversationState {
    return {
      ...state,
      additionalInstructions: state.additionalInstructions
        ? [...state.additionalInstructions]
        : undefined,
      agent: state.agent
        ? {
            ...state.agent,
            messages: state.agent.messages
              ? [...state.agent.messages]
              : undefined,
            chatLog: state.agent.chatLog
              ? state.agent.chatLog.map((entry) => ({ ...entry }))
              : undefined,
            outputData: state.agent.outputData
              ? { ...state.agent.outputData }
              : undefined,
            lastVerification: state.agent.lastVerification
              ? { ...state.agent.lastVerification }
              : undefined,
          }
        : undefined,
      pipeline: state.pipeline ? { ...state.pipeline } : undefined,
      metadata: state.metadata ? { ...state.metadata } : undefined,
    };
  }

  private stringifyInstruction(message: unknown): string | undefined {
    if (typeof message === "string") {
      const trimmed = message.trim();
      return trimmed.length ? trimmed : undefined;
    }

    if (typeof message === "number" || typeof message === "boolean") {
      return String(message);
    }

    if (!message) {
      return undefined;
    }

    try {
      return JSON.stringify(message);
    } catch {
      return undefined;
    }
  }

  private applyControlRequests(
    state: PersistedConversationState,
    requests?: AiAgentControlRequest[]
  ): PersistedConversationState {
    if (!requests?.length) {
      return state;
    }

    const next = state;
    for (const request of requests) {
      switch (request.type) {
        case "pause": {
          if (next.phase !== "paused") {
            next.resumePhase = next.phase;
          }
          next.phase = "paused";
          next.pausedReason = request.reason;
          break;
        }
        case "resume": {
          if (next.phase === "paused") {
            next.phase = next.resumePhase ?? "idle";
            next.resumePhase = undefined;
            next.pausedReason = undefined;
          }
          break;
        }
        case "addMessages": {
          const additions = request.messages
            .map((message) => this.stringifyInstruction(message))
            .filter(
              (value): value is string =>
                typeof value === "string" && value.length > 0
            );
          if (!additions.length) {
            break;
          }
          next.additionalInstructions = [
            ...(next.additionalInstructions ?? []),
            ...additions,
          ];
          next.agent = undefined;
          next.pipeline = undefined;
          next.phase = "idle";
          next.metadata = {
            ...(next.metadata ?? {}),
            lastInstructionAt: Date.now(),
          };
          break;
        }
        default:
          break;
      }
    }

    return next;
  }

  private composePrompt(state: PersistedConversationState): string {
    if (!state.additionalInstructions?.length) {
      return state.prompt;
    }

    const instructions = state.additionalInstructions
      .map((instruction, index) => `${index + 1}. ${instruction}`)
      .join("\n");

    return `${state.prompt}\n\nAdditional instructions:\n${instructions}`;
  }

  private describePhase(state: PersistedConversationState): string {
    switch (state.phase) {
      case "idle":
        return "Idle";
      case "acting":
        return "Working";
      case "awaiting-user":
        return state.agent?.pendingQuestion
          ? `Awaiting user: ${state.agent.pendingQuestion}`
          : "Awaiting user";
      case "complete":
        return "Complete";
      case "paused":
        return state.pausedReason ? `Paused: ${state.pausedReason}` : "Paused";
      case "error":
        return "Error";
      default:
        return "Unknown";
    }
  }

  private buildAgentState(
    conversationId: string,
    state: PersistedConversationState,
    statusOverride?: string,
    isCompleteOverride?: boolean
  ): AiAgentConversationState<PersistedConversationState> {
    return {
      id: conversationId,
      state,
      status: statusOverride ?? this.describePhase(state),
      metadata: state.metadata,
      isPaused: state.phase === "paused",
      isComplete:
        isCompleteOverride ??
        (state.phase === "complete" && !state.pausedReason),
    };
  }

  private normalizeValueForHash(value: unknown): unknown {
    if (Array.isArray(value)) {
      return value.map((item) => this.normalizeValueForHash(item));
    }

    if (value && typeof value === "object") {
      return Object.keys(value as Record<string, unknown>)
        .sort()
        .reduce<Record<string, unknown>>((acc, key) => {
          acc[key] = this.normalizeValueForHash(
            (value as Record<string, unknown>)[key]
          );
          return acc;
        }, {});
    }

    return value;
  }

  private stableHash(value: unknown): string {
    return JSON.stringify(this.normalizeValueForHash(value));
  }

  private arePipelinesEqual(
    first?: Record<string, GenericNode>,
    second?: Record<string, GenericNode>
  ): boolean {
    if (!first && !second) {
      return true;
    }

    if (!first || !second) {
      return false;
    }

    return this.stableHash(first) === this.stableHash(second);
  }

  private async executePromptWorkflow({
    page,
    conversationId,
    state,
    previousPipeline,
    callbacks,
    controlRequests,
    abortController,
  }: PromptWorkflowParams): Promise<
    AiAgentConversationState<PersistedConversationState>
  > {
    let nextState = this.cloneState(state);
    nextState = this.applyControlRequests(nextState, controlRequests);

    const promptText = this.composePrompt(nextState);

    if (nextState.phase === "paused") {
      await callbacks.onStatusUpdate?.(
        nextState.pausedReason ? `Paused: ${nextState.pausedReason}` : "Paused"
      );
      return this.buildAgentState(conversationId, nextState);
    }

    if (nextState.phase === "awaiting-user") {
      await callbacks.onStatusUpdate?.(
        nextState.agent?.pendingQuestion
          ? `Awaiting user input: ${nextState.agent.pendingQuestion}`
          : "Awaiting user input"
      );
      return this.buildAgentState(
        conversationId,
        nextState,
        this.describePhase(nextState),
        false
      );
    }

    await callbacks.onStatusUpdate?.("Agent working");
    nextState.phase = "acting";
    const agentResult = await this.runAgentInternal(
      page,
      promptText,
      previousPipeline,
      callbacks,
      abortController
    );

    nextState.agent = {
      messages: agentResult.messages,
      lastMessage: agentResult.outputText,
      lastTool: agentResult.lastTool,
      pendingQuestion: agentResult.pendingQuestion,
      outputData: agentResult.outputData
        ? { ...agentResult.outputData }
        : undefined,
      chatLog: agentResult.chatMessages.length
        ? [...agentResult.chatMessages]
        : undefined,
      lastVerification: agentResult.verification
        ? { ...agentResult.verification }
        : undefined,
    };

    nextState.metadata = {
      ...(nextState.metadata ?? {}),
      lastAgentMessage: agentResult.outputText,
      lastAgentTool: agentResult.lastTool,
      pendingQuestion: agentResult.pendingQuestion,
      lastOutputDataSummary: agentResult.outputData?.description,
      lastUpdatedAt: Date.now(),
    };

    for (const chat of agentResult.chatMessages) {
      await callbacks.onStatusUpdate?.(
        `[chat:${chat.audience}] ${chat.message}`
      );
    }

    if (agentResult.outputData) {
      const prefix = agentResult.outputData.final ? "Output" : "Progress";
      await callbacks.onStatusUpdate?.(
        `${prefix}: ${agentResult.outputData.description ?? "Data ready"}`
      );
    }

    if (agentResult.pendingQuestion) {
      nextState.phase = "awaiting-user";
      await callbacks.onStatusUpdate?.(
        `Awaiting user input: ${agentResult.pendingQuestion}`
      );
      return this.buildAgentState(
        conversationId,
        nextState,
        "Awaiting user input",
        false
      );
    }

    if (agentResult.emittedPipeline) {
      const pipelineChanged = !this.arePipelinesEqual(
        previousPipeline,
        agentResult.emittedPipeline.raw
      );

      nextState.pipeline = {
        json: agentResult.emittedPipeline.raw,
        messages: agentResult.messages,
        updatedAt: Date.now(),
      };
      nextState.metadata = {
        ...(nextState.metadata ?? {}),
        pipelineChanged,
        pipelineUpdatedAt: nextState.pipeline.updatedAt,
      };
      nextState.phase = "complete";

      if (pipelineChanged) {
        await callbacks.onPipelineUpdate(agentResult.emittedPipeline.pipeline);
        await callbacks.onStatusUpdate?.("Pipeline updated");
        return this.buildAgentState(
          conversationId,
          nextState,
          "Pipeline updated",
          true
        );
      }

      await callbacks.onStatusUpdate?.("Pipeline unchanged");
      return this.buildAgentState(
        conversationId,
        nextState,
        "Pipeline unchanged",
        true
      );
    }

    if (agentResult.outputData?.final) {
      nextState.phase = "complete";
      nextState.metadata = {
        ...(nextState.metadata ?? {}),
        outputCompletedAt: Date.now(),
      };
      return this.buildAgentState(
        conversationId,
        nextState,
        agentResult.outputData.description ?? "Output ready",
        true
      );
    }

    if (agentResult.finishRequested === false) {
      nextState.phase = "idle";
      return this.buildAgentState(
        conversationId,
        nextState,
        agentResult.outputText ?? "Idle",
        false
      );
    }

    nextState.phase = "complete";
    return this.buildAgentState(
      conversationId,
      nextState,
      agentResult.outputText ?? "Complete",
      true
    );
  }

  createModel<TModelContext>(
    model: string,
    context: TModelContext
  ): AiModel<TModelContext> {
    return new OpenAiModel<TModelContext>(
      this.config.logger.createScope(model),
      model,
      this.config,
      context,
      {
        emitStartChecked: (base) => this.emitStartChecked(base),
        emitEndChecked: (base, started) => this.emitEndChecked(base, started),
      }
    );
  }

  constructor(
    config: OpenAiConfig,
    functionCall: (
      ctx: Page,
      name: string,
      params: any,
      abortController?: AbortController
    ) => Promise<unknown>,
    monitor?: UsageMonitor
  ) {
    super(
      { ...config, logger: config.logger.createScope("openai") },
      functionCall,
      monitor
    );

    const agentModel =
      config.models?.agent ??
      config.models?.generate ??
      config.models?.analyze ??
      PIPELINE_GENERATE_MODEL;
    assert(
      OpenAiProvider.models.agent.includes(agentModel),
      "Invalid agent model"
    );
    this.agentModel = agentModel;

    if (config.models?.status) {
      assert(
        OpenAiProvider.models.status.includes(config.models.status),
        "Invalid status model"
      );
      this.statusModel = config.models.status;
    } else {
      this.statusModel =
        config.models?.analyze ??
        config.models?.generate ??
        config.models?.agent ??
        OpenAiProvider.models.status[0];
    }
  }

  static get models() {
    const supported = [
      "gpt-4o-mini",
      "gpt-5-nano",
      "gpt-5-mini",
      "gpt-5",
      "gpt-5.1",
    ];
    return {
      generic: supported,
      analyze: supported,
      generate: supported,
      agent: supported,
      status: supported,
    };
  }

  private truncateStatusSnippet(text: string, limit = 400): string {
    if (text.length <= limit) {
      return text;
    }
    return `${text.slice(0, limit)}...`;
  }

  private describeValueForStatus(value: unknown): string | undefined {
    if (value === undefined || value === null) {
      return undefined;
    }

    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed ? this.truncateStatusSnippet(trimmed) : undefined;
    }

    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }

    if (Array.isArray(value)) {
      const serialized = (() => {
        try {
          return JSON.stringify(value);
        } catch {
          return undefined;
        }
      })();
      const summary = serialized
        ? this.truncateStatusSnippet(serialized)
        : `${value.length} items`;
      return `Array(len=${value.length}) ${summary}`;
    }

    if (typeof value === "object") {
      try {
        const json = JSON.stringify(value);
        if (!json || json === "{}") {
          return undefined;
        }
        return this.truncateStatusSnippet(json);
      } catch {
        return undefined;
      }
    }

    return undefined;
  }

  private async createFunctionStatusUpdate(
    phase: "intent" | "success" | "failure",
    name: string,
    params: unknown,
    details?: { result?: unknown; error?: unknown },
    abortController?: AbortController
  ): Promise<string | undefined> {
    try {
      const inputLines = [
        `Phase: ${phase}`,
        `Function: ${name}`,
        `Parameters: ${
          this.describeValueForStatus(params) ?? "Not provided"
        }`,
      ];

      if ("result" in (details ?? {}) || phase === "success") {
        inputLines.push(
          `Result: ${
            this.describeValueForStatus(details?.result) ?? "No structured data"
          }`
        );
      }

      if ("error" in (details ?? {}) || phase === "failure") {
        inputLines.push(
          `Error: ${
            this.describeValueForStatus(details?.error) ?? "No error details"
          }`
        );
      }

      const response = await (
        await this.config.openAi
      ).responses.create(
        {
          model: this.statusModel,
          temperature: 0.2,
          input: [
            {
              role: "system",
              content:
                "You craft short, natural status updates describing browser automation tool calls. " +
                "Use present-progressive verbs for intent, past tense for success, and actionable phrasing for failures. " +
                "Stay under 18 words, avoid quoting function names, and speak as the agent narrating its work.",
            },
            {
              role: "user",
              content: inputLines.join("\n"),
            },
          ],
          text: {
            format: {
              type: "text",
            },
          },
        },
        abortController ? { signal: abortController.signal } : undefined
      );

      const text = response.output_text?.trim();
      return text || undefined;
    } catch (error) {
      this.config.logger.log(
        "warn",
        "status",
        "Failed to generate function status update",
        {
          phase,
          function: name,
          error: createErrorObject(error),
        }
      );
      return undefined;
    }
  }

  protected override async callFunctionWithTelemetry(
    context: Page,
    name: string,
    params: any,
    options?: FunctionCallStatusOptions
  ): Promise<unknown> {
    if (!options?.onStatusUpdate) {
      return super.callFunctionWithTelemetry(context, name, params, options);
    }

    const abortController = options.abortController;
    const safeStatusUpdate = async (message?: string) => {
      if (!message) {
        return;
      }
      try {
        await options.onStatusUpdate?.(message);
      } catch {
        // Ignore downstream status update failures.
      }
    };

    const intentStatus = await this.createFunctionStatusUpdate(
      "intent",
      name,
      params,
      undefined,
      abortController
    );

    if (!intentStatus) {
      return super.callFunctionWithTelemetry(context, name, params, options);
    }

    await safeStatusUpdate(intentStatus);

    const telemetryOptions: FunctionCallStatusOptions | undefined = options
      ? { ...options, onStatusUpdate: undefined, describe: undefined }
      : undefined;

    try {
      const result = await super.callFunctionWithTelemetry(
        context,
        name,
        params,
        telemetryOptions
      );
      const successStatus = await this.createFunctionStatusUpdate(
        "success",
        name,
        params,
        { result },
        abortController
      );
      await safeStatusUpdate(successStatus);
      return result;
    } catch (error) {
      const failureStatus = await this.createFunctionStatusUpdate(
        "failure",
        name,
        params,
        {
          error:
            error instanceof Error ? { message: error.message } : (error as unknown),
        },
        abortController
      );
      await safeStatusUpdate(failureStatus);
      throw error;
    }
  }

  private async runAgentInternal(
    page: Page,
    userPrompt: string,
    previousPipeline: Record<string, GenericNode>,
    callbacks: PipelineConversationCallbacks,
    abortController?: AbortController
  ): Promise<AgentRunResult> {
    this.config.logger.log("info", "agent", "Unified agent started", {
      prompt: userPrompt,
    });
    this.throwIfAborted(abortController);

    const functionPromptSections = await this.getFunctionPromptSections();
    this.throwIfAborted(abortController);

    const promptConfig: ModelConfiguration = {
      model: this.agentModel,
      prompt: [
        {
          role: "system",
          content: agentPrompt({ functions: functionPromptSections }),
        },
        {
          role: "system",
          content: `Previous pipeline version:\n ${JSON.stringify(
            previousPipeline
          )}`,
        },
        {
          role: "user",
          content: `User request:\n ${userPrompt}`,
        },
      ],
      text: {
        format: {
          type: "text",
        },
      },
    };

    const start = await this.emitStartChecked({
      source: "agent.unified",
      model: promptConfig.model,
      metadata: {
        promptLength: userPrompt.length,
      },
    });

    if (!start.proceed) {
      this.config.logger.log("debug", "agent", "Monitor vetoed start", {
        model: promptConfig.model,
      });
      return { messages: [], chatMessages: [] };
    }

    const functionCallRef = (
      fnContext: Page,
      name: string,
      params: unknown,
      controller?: AbortController
    ) =>
      this.callFunctionWithTelemetry(fnContext, name, params, {
        abortController: controller,
        onStatusUpdate: callbacks.onStatusUpdate,
      });
    const self = this;
    const browserToolNames = new Set(analyzeTools.map((tool) => tool.name));
    const workingState: {
      emittedPipeline?: EmittedPipelineResult;
      pendingQuestion?: string;
      outputData?: AgentOutputData;
      chatMessages: AgentChatMessage[];
      verification?: AgentRunVerification;
      finishRequested?: boolean;
      lastTool?: string;
    } = {
      chatMessages: [],
    };

    const validateAndCompilePipeline = (
      candidate: unknown
    ):
      | { success: true; raw: Record<string, GenericNode>; pipeline: Pipeline }
      | { success: false; errors: unknown[] } => {
      let normalized: unknown = candidate;
      if (typeof normalized === "string") {
        const trimmed = normalized.trim();
        if (!trimmed) {
          throw new Error("pipeline is required");
        }
        try {
          normalized = JSON.parse(trimmed);
        } catch (error) {
          throw new Error(
            `pipeline must be valid JSON: ${(error as Error).message}`
          );
        }
      }

      if (!normalized || typeof normalized !== "object" || Array.isArray(normalized)) {
        throw new Error("pipeline must be a JSON object");
      }

      const pipelineJson = normalized as Record<string, GenericNode>;
      if (!validatePipelineSchema(pipelineJson)) {
        const errs = getPipelineValidationErrors() ?? [];
        return { success: false, errors: errs };
      }

      const compiled = this.config.pipelineProvider.compile(pipelineJson);
      if (!hasPipeline(compiled)) {
        return { success: false, errors: compiled.errors };
      }

      return { success: true, raw: pipelineJson, pipeline: compiled.pipeline! };
    };

    const pipelineToolParameters = createPipelineToolSchema();

    const jsonValueDefinition = {
      description:
        "Represents any JSON value, including deeply nested structures.",
      anyOf: [
        {
          type: "object",
          additionalProperties: false,
          patternProperties: {
            "^.+$": { $ref: "#/$defs/jsonValue" },
          },
        },
        {
          type: "array",
          items: { $ref: "#/$defs/jsonValue" },
        },
        { type: "string" },
        { type: "number" },
        { type: "boolean" },
        { type: "null" },
      ],
    };

    const providerTools: OpenAI.Responses.FunctionTool[] = [
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
      {
        type: "function",
        name: "getPreviousPipeline",
        description: "Returns the last emitted pipeline, if any.",
        parameters: {
          type: "object",
          properties: {},
          required: [],
          additionalProperties: false,
        },
        strict: true,
      },
      {
        type: "function",
        name: "compilePipeline",
        description:
          "Validate a pipeline draft without emitting it. Use before running or emitting.",
        parameters: pipelineToolParameters,
        strict: true,
      },
      {
        type: "function",
        name: "runPipeline",
        description:
          "Dry-run a pipeline to verify structure. This checks compilation and dependencies.",
        parameters: pipelineToolParameters,
        strict: true,
      },
      {
        type: "function",
        name: "emitPipeline",
        description:
          "Emit a validated pipeline to the user. Only call when it is production-ready.",
        parameters: pipelineToolParameters,
        strict: true,
      },
      {
        type: "function",
        name: "requestUserInput",
        description:
          "Ask the user for more information when you cannot proceed safely.",
        parameters: {
          type: "object",
          properties: {
            question: { type: "string" },
          },
          patternProperties: {
            "^urgency$": {
              type: "string",
              enum: ["blocking", "optional"],
              description: "Describe whether the task is blocked.",
            },
          },
          required: ["question"],
          additionalProperties: false,
        },
        strict: true,
      },
      {
        type: "function",
        name: "provideOutputData",
        description:
          "Share structured or textual output with the user when no pipeline is required.",
        parameters: {
          type: "object",
          properties: {
            data: { $ref: "#/$defs/jsonValue" },
          },
          required: ["data"],
          additionalProperties: false,
          patternProperties: {
            "^description$": {
              type: "string",
              description: "Short summary of the output.",
            },
            "^final$": {
              type: "boolean",
              description:
                "Set true if this output fulfills the user's request completely.",
            },
          },
          $defs: {
            jsonValue: jsonValueDefinition,
          },
        },
        strict: true,
      },
      {
        type: "function",
        name: "chatWithUser",
        description:
          "Send a conversational message to the user or other observers without finishing the task.",
        parameters: {
          type: "object",
          properties: {
            message: { type: "string" },
          },
          required: ["message"],
          additionalProperties: false,
          patternProperties: {
            "^audience$": {
              type: "string",
              enum: ["user", "observers"],
              description: "Who should see this message?",
            },
          },
        },
        strict: true,
      },
    ];

    const functionsConfig: FunctionConfiguration = {
      tools: [...analyzeTools, ...providerTools],
      async handle(call, _history, handlerAbortController) {
        const params = call.arguments?.length ? JSON.parse(call.arguments) : {};
        workingState.lastTool = call.name;

        if (browserToolNames.has(call.name)) {
          const result = await functionCallRef(
            page,
            call.name,
            params,
            handlerAbortController ?? abortController
          );
          return result ?? {};
        }

        switch (call.name) {
          case "getNodeSchema": {
            const nodeName = params.node as string;
            const builtinSchema = getBuiltinNodeSchema(
              nodeName
            ) as unknown as Record<string, unknown> | undefined;
            const nodeSchema =
              builtinSchema ?? (await self.getFunctionNodeSchema(nodeName));
            if (!nodeSchema) {
              throw new TypeError(
                `Node schema for "${nodeName}" was not found`
              );
            }
            return nodeSchema;
          }
          case "getPreviousPipeline": {
            return previousPipeline;
          }
          case "compilePipeline": {
            const result = validateAndCompilePipeline(params.pipeline);
            if (!result.success) {
              workingState.verification = {
                success: false,
                errors: result.errors,
                note: params.reason ?? "compile",
                timestamp: Date.now(),
              };
              return { success: false, errors: result.errors };
            }
            workingState.verification = {
              success: true,
              note: params.reason ?? "compile",
              timestamp: Date.now(),
            };
            return {
              success: true,
              nodeCount: Object.keys(result.raw).length,
            };
          }
          case "runPipeline": {
            const result = validateAndCompilePipeline(params.pipeline);
            if (!result.success) {
              workingState.verification = {
                success: false,
                errors: result.errors,
                note: params.reason ?? "run",
                timestamp: Date.now(),
              };
              return { success: false, errors: result.errors };
            }
            workingState.verification = {
              success: true,
              note: params.reason ?? "run",
              timestamp: Date.now(),
            };
            return {
              success: true,
              note: "Dry-run successful (structure verified)",
            };
          }
          case "emitPipeline": {
            const result = validateAndCompilePipeline(params.pipeline);
            if (!result.success) {
              return { success: false, errors: result.errors };
            }
            workingState.emittedPipeline = {
              raw: result.raw,
              pipeline: result.pipeline,
            };
            workingState.finishRequested =
              params.final !== undefined ? Boolean(params.final) : true;
            return {
              success: true,
              nodeCount: Object.keys(result.raw).length,
            };
          }
          case "requestUserInput": {
            const rawQuestion = String(params.question ?? "").trim();
            if (!rawQuestion) {
              throw new Error("question is required");
            }
            workingState.pendingQuestion = rawQuestion;
            workingState.finishRequested = false;
            return { acknowledged: true };
          }
          case "provideOutputData": {
            workingState.outputData = {
              description:
                typeof params.description === "string"
                  ? params.description
                  : undefined,
              data: params.data,
              final: Boolean(params.final),
            };
            if (params.final) {
              workingState.finishRequested = true;
            }
            return { recorded: true };
          }
          case "chatWithUser": {
            workingState.chatMessages.push({
              message: String(params.message ?? ""),
              audience: params.audience === "observers" ? "observers" : "user",
              at: Date.now(),
            });
            return { acknowledged: true };
          }
          default:
            throw new Error(`Unsupported tool call: ${call.name}`);
        }
      },
    };

    const convo = new Conversation({
      model: promptConfig,
      functions: {
        ...functionsConfig,
        handle: functionsConfig.handle.bind(this),
      },
      openAi: this.config.openAi,
      onMessages: callbacks.onMessages,
      logger: this.config.logger.createScope(promptConfig.model),
      abortController,
    });

    try {
      for await (const _ of convo.start()) {
        this.throwIfAborted(abortController);
      }

      const end = await this.emitEndChecked(
        {
          source: "agent.unified",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
          usage: convo.usage,
        },
        start.event.startedAt
      );

      if (!end.proceed) {
        this.config.logger.log("debug", "agent", "Monitor vetoed end", {
          model: promptConfig.model,
        });
      }

      return {
        messages: convo.history,
        outputText: convo.output ?? undefined,
        emittedPipeline: workingState.emittedPipeline,
        pendingQuestion: workingState.pendingQuestion,
        outputData: workingState.outputData,
        chatMessages: workingState.chatMessages,
        verification: workingState.verification,
        finishRequested: workingState.finishRequested,
        lastTool: workingState.lastTool,
      };
    } catch (error) {
      await this.emitEndChecked(
        {
          source: "agent.unified",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
        },
        start.event.startedAt
      );
      throw error;
    }
  }

  public async prompt(
    page: Page,
    params: PromptParams
  ): Promise<AiAgentConversationState<PersistedConversationState>> {
    const state = this.createInitialState(params.userPrompt);
    return this.executePromptWorkflow({
      page,
      conversationId: randomUUID(),
      state,
      previousPipeline: params.previousPipeline,
      callbacks: params,
      controlRequests: params.controlRequests,
      abortController: params.abortController,
    });
  }

  public async continuePrompt(
    page: Page,
    params: ContinuePromptParams
  ): Promise<AiAgentConversationState<PersistedConversationState>> {
    const conversationId = params.conversation.id ?? randomUUID();
    const state = this.normalizeState(params.conversation.state);
    return this.executePromptWorkflow({
      page,
      conversationId,
      state,
      previousPipeline: params.previousPipeline,
      callbacks: params,
      controlRequests: params.controlRequests,
      abortController: params.abortController,
    });
  }
}
