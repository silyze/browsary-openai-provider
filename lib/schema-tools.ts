import { genericNodeSchema } from "@silyze/browsary-pipeline";

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
