import OpenAI from "openai";
import { MAX_RESPONSE_FIX_RETRY } from "./defaults";
import { assert } from "@mojsoski/assert";
import {
  AiModel,
  AiProvider,
  AiResult,
  AnalysisResult,
  UsageMonitor,
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
import {
  buildFunctionPromptSections,
  FunctionPromptSections,
} from "./prompts/functions";
import { Page } from "puppeteer-core";
import { type Logger } from "@silyze/logger";
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

export type OpenAiConfig = {
  openAi: OpenAI;
  pipelineProvider: PipelineProvider;
  functionProvider?: PipelineFunctionProvider;
  logger: Logger;
  models?: {
    analyze?: string;
    generate?: string;
  };
};

export class OpenAiProvider extends AiProvider<Page, OpenAiConfig> {
  private functionPromptSectionsPromise?: Promise<
    FunctionPromptSections | undefined
  >;

  private throwIfAborted(abortController?: AbortController) {
    if (!abortController?.signal.aborted) {
      return;
    }

    const reason = abortController.signal.reason;
    if (reason instanceof Error) {
      throw reason;
    }

    throw new Error(
      typeof reason === "string" ? reason : "Operation aborted"
    );
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
    onMessages?: (msgs: unknown[]) => void | Promise<void>,
    abortController?: AbortController
  ): Promise<AiResult<AnalysisResult>> {
    this.config.logger.log("info", "analyze", "Analysis started", {
      prompt: userPrompt,
    });
    this.throwIfAborted(abortController);

    const functionPromptSections = await this.getFunctionPromptSections();
    this.throwIfAborted(abortController);

    const promptConfig: ModelConfiguration = {
      model: this.config.models?.analyze ?? ANALYZE_MODEL,
      prompt: [
        {
          role: "system",
          content: analyzePrompt({ functions: functionPromptSections }),
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

    const start = await this.emitStartChecked({
      source: "pipeline.analyze",
      model: promptConfig.model,
      metadata: {
        promptLength: userPrompt.length,
      },
    });

    if (!start.proceed) {
      this.config.logger.log("debug", "analyze", "Monitor vetoed start", {
        model: promptConfig.model,
      });
      return { messages: [] };
    }

    const functionCallRef = this.callFunctionWithTelemetry.bind(this);
    const functionsConfig: FunctionConfiguration = {
      tools: analyzeTools,
      async handle(call, _history, handlerAbortController) {
        const params = JSON.parse(call.arguments);
        const result = await functionCallRef(
          page,
          call.name,
          params,
          handlerAbortController ?? abortController
        );
        return result ?? {};
      },
    };

    const convo = new Conversation({
      model: promptConfig,
      functions: functionsConfig,
      openAi: this.config.openAi,
      onMessages,
      logger: this.config.logger.createScope(promptConfig.model),
      abortController,
    });

    try {
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

        const end = await this.emitEndChecked(
          {
            source: "pipeline.analyze",
            model: promptConfig.model,
            startedAt: start.event.startedAt,
            metadata: { promptLength: userPrompt.length },
            usage: convo.usage,
          },
          start.event.startedAt
        );

        if (!end.proceed) {
          this.config.logger.log("debug", "analyze", "Monitor vetoed end", {
            model: promptConfig.model,
          });
          return { messages };
        }

        return { messages };
      }

      const analysis = JSON.parse(output) as AnalysisResult["analysis"];
      this.config.logger.log("info", "analyze", "Analysis ended", {
        analysis,
        prompt: userPrompt,
      });

      const end = await this.emitEndChecked(
        {
          source: "pipeline.analyze",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
          usage: convo.usage,
        },
        start.event.startedAt
      );

      if (!end.proceed) {
        this.config.logger.log("debug", "analyze", "Monitor vetoed end", {
          model: promptConfig.model,
        });
        return { messages };
      }

      return { result: { analysis, prompt: userPrompt }, messages };
    } catch (error) {
      await this.emitEndChecked(
        {
          source: "pipeline.analyze",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
          usage: convo.usage,
        },
        start.event.startedAt
      );
      throw error;
    }
  }

  public async generate(
    page: Page,
    { analysis, prompt: userPrompt }: AnalysisResult,
    previousPipeline: Record<string, GenericNode>,
    onMessages?: (msgs: unknown[]) => void | Promise<void>,
    abortController?: AbortController
  ): Promise<AiResult<Pipeline>> {
    let retryCount = 0;
    this.throwIfAborted(abortController);

    const functionPromptSections = await this.getFunctionPromptSections();
    this.throwIfAborted(abortController);

    const initialPrompt = pipelinePrompt(analysis, functionPromptSections);
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

    const start = await this.emitStartChecked({
      source: "pipeline.generate",
      model: promptConfig.model,
      metadata: {
        promptLength: userPrompt.length,
      },
    });

    if (!start.proceed) {
      this.config.logger.log("debug", "generate", "Monitor vetoed start", {
        model: promptConfig.model,
      });
      return { messages: [] };
    }

    let convo: Conversation | undefined;

    const self = this;
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
      async handle(call, history, _handlerAbortController) {
        const params = JSON.parse(call.arguments);
        const nodeName = params.node as string;
        const builtinSchema = pipelineSchema.additionalProperties.anyOf.find(
          (s) => s.properties.node.const === nodeName
        ) as unknown as Record<string, unknown> | undefined;
        const nodeSchema =
          builtinSchema ?? (await self.getFunctionNodeSchema(nodeName));
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
      abortController,
    });

    try {
      while (retryCount < MAX_RESPONSE_FIX_RETRY) {
        this.throwIfAborted(abortController);
        for await (const _ of convo.start()) {
        }

        const output = convo.output;
        if (output) {
          try {
            const parsed = JSON.parse(output);
            if (!validatePipelineSchema(parsed)) {
              const errs = getPipelineValidationErrors();
              throw new Error(
                `Validation errors: ${errs
                  ?.map((item) => JSON.stringify(item))
                  .join(", ")}`
              );
            }
            const compiled = this.config.pipelineProvider.compile(parsed);
            if (!hasPipeline(compiled)) {
              throw new Error(
                `Compile errors: ${compiled.errors
                  .map((item) => JSON.stringify(item))
                  .join(", ")}`
              );
            }

            const end = await this.emitEndChecked(
              {
                source: "pipeline.generate",
                model: promptConfig.model,
                startedAt: start.event.startedAt,
                metadata: { promptLength: userPrompt.length },
                usage: convo.usage,
              },
              start.event.startedAt
            );

            if (!end.proceed) {
              this.config.logger.log(
                "debug",
                "generate",
                "Monitor vetoed end",
                { model: promptConfig.model }
              );
              return { messages: convo.history };
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

      const end = await this.emitEndChecked(
        {
          source: "pipeline.generate",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
          usage: convo?.usage,
        },
        start.event.startedAt
      );

      if (!end.proceed) {
        this.config.logger.log("debug", "generate", "Monitor vetoed end", {
          model: promptConfig.model,
        });
      }

      return { messages: convo?.history ?? [] };
    } catch (error) {
      await this.emitEndChecked(
        {
          source: "pipeline.generate",
          model: promptConfig.model,
          startedAt: start.event.startedAt,
          metadata: { promptLength: userPrompt.length },
          usage: convo?.usage,
        },
        start.event.startedAt
      );
      throw error;
    }
  }
}
