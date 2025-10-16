import {
  PipelineFunctionDescriptor,
  PipelineFunctionParameter,
  PipelineFunctionProvider,
} from "@silyze/browsary-pipeline";

export type FunctionPromptSections = {
  index: string;
  example?: string;
};

function formatParameters(
  parameters: PipelineFunctionParameter[]
): string | undefined {
  if (!parameters.length) {
    return undefined;
  }

  const entries = parameters.map((param) => {
    const suffix = param.description ? ` (${param.description})` : "";
    return `${param.name}: ${param.refType}${suffix}`;
  });

  return entries.join(", ");
}

function formatDescriptor(fn: PipelineFunctionDescriptor): string {
  const identifier = `${fn.namespace}::${fn.name}`;
  const metaParts = [fn.metadata?.title, fn.metadata?.description].filter(
    Boolean
  );

  const ioParts: string[] = [];

  const formattedInputs = formatParameters(fn.inputs);
  if (formattedInputs) {
    ioParts.push(`**Inputs**: \`{ ${formattedInputs} }\``);
  }

  const formattedOutputs = formatParameters(
    fn.outputs.map((output) => ({
      name: output.name,
      refType: output.refType,
      description: output.description,
    }))
  );

  if (formattedOutputs) {
    ioParts.push(`**Outputs**: \`{ ${formattedOutputs} }\``);
  }

  const detailParts = [...metaParts];
  if (ioParts.length) {
    detailParts.push(ioParts.join(". "));
  }

  const detail = detailParts.length ? ` - ${detailParts.join(". ")}` : "";

  return `- \`${identifier}\`${detail}`;
}

function sanitizeSegment(value: string): string {
  const sanitized = value
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return sanitized.length ? sanitized : "function";
}

function buildExample(fn: PipelineFunctionDescriptor): string {
  const identifier = `${fn.namespace}::${fn.name}`;
  const stepBase = `call_${sanitizeSegment(fn.namespace)}_${sanitizeSegment(
    fn.name
  )}`;

  const argsValue = Object.fromEntries(
    fn.inputs.map((input) => [input.name, `<${input.refType}>`])
  );

  const outputs: Record<string, string> = {};
  for (const output of fn.outputs) {
    const key = `${stepBase}_${sanitizeSegment(output.name)}`;
    outputs[output.name] = key;
  }

  if (!("result" in outputs)) {
    outputs.result = `${stepBase}_result`;
  }

  const exampleNode = {
    [stepBase]: {
      node: "functions::call",
      inputs: {
        identifier: { type: "constant", value: identifier },
        args: {
          type: "constant",
          value: argsValue,
        },
      },
      outputs,
      dependsOn: [],
    },
  };

  const exampleJson = JSON.stringify(exampleNode, null, 2);

  return `## Function call example\nUse \`functions::call\` to execute a reusable function.\n\`\`\`json\n${exampleJson}\n\`\`\``;
}

export async function buildFunctionPromptSections(
  provider?: PipelineFunctionProvider
): Promise<FunctionPromptSections | undefined> {
  if (!provider) {
    return undefined;
  }

  const namespaces = await provider.listNamespaces();
  const descriptors: PipelineFunctionDescriptor[] = [];

  const sortedNamespaces = [...namespaces].sort((a, b) => a.localeCompare(b));

  for (const namespace of sortedNamespaces) {
    const functions = await provider.listFunctions(namespace);
    functions.sort((a, b) => a.name.localeCompare(b.name));
    descriptors.push(...functions);
  }

  if (!descriptors.length) {
    return undefined;
  }

  const indexLines = descriptors.map(formatDescriptor).join("\n");
  const index = `## Available functions:\n${indexLines}`;

  const example = buildExample(descriptors[0]);

  return {
    index,
    example,
  };
}
