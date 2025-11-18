import {
  genericNodeSchema,
  pipelineSchema,
} from "@silyze/browsary-pipeline";

const pipelineStructureSchema = {
  type: "object",
  properties: {},
  additionalProperties: {
    $ref: "#/$defs/GenericNode",
  },
} as const;

export type PipelineToolSchema = ReturnType<typeof createPipelineToolSchema>;

export function createPipelineToolSchema() {
  return {
    type: "object",
    properties: {
      pipeline: pipelineStructureSchema,
      label: {
        type: "string",
        description: "Optional label to describe the attempt.",
      },
      final: {
        type: "boolean",
        description:
          "Set true to mark the pipeline/output as ready for delivery.",
      },
      reason: {
        type: "string",
        description: "Optional reason or summary for the action.",
      },
    },
    required: ["pipeline"],
    additionalProperties: false,
    $defs: genericNodeSchema.$defs,
  } as const;
}

export { pipelineStructureSchema };

export type PipelineNodeSchema = {
  description?: string;
  properties: {
    node: { const: string };
    inputs?: { properties?: Record<string, unknown> };
    outputs?: { properties?: Record<string, unknown> };
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

type AnyStandardNode =
  (typeof pipelineSchema.additionalProperties)["anyOf"][number];

function hasNodeProperties(
  candidate: AnyStandardNode
): candidate is AnyStandardNode & PipelineNodeSchema {
  return (
    typeof candidate === "object" &&
    candidate !== null &&
    "properties" in candidate &&
    typeof (
      candidate as { properties?: Record<string, unknown> }
    ).properties === "object" &&
    typeof (
      (candidate as {
        properties?: { node?: { const?: string } };
      }).properties?.node?.const ?? ""
    ) === "string"
  );
}

export function getBuiltinNodeSchema(
  nodeName: string,
  schema: typeof pipelineSchema = pipelineSchema
): PipelineNodeSchema | undefined {
  return schema.additionalProperties.anyOf.find(
    (def): def is AnyStandardNode & PipelineNodeSchema =>
      hasNodeProperties(def) && def.properties.node.const === nodeName
  ) as PipelineNodeSchema | undefined;
}

export function getAllBuiltinNodeSchemas(
  schema: typeof pipelineSchema = pipelineSchema
): PipelineNodeSchema[] {
  return schema.additionalProperties.anyOf
    .filter(hasNodeProperties)
    .map((node) => node as PipelineNodeSchema);
}
