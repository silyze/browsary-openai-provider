import OpenAI from "openai";
import { Logger } from "@silyze/logger";

type ResponsePayload = OpenAI.Responses.ResponseCreateParamsNonStreaming;
type ResponseOptions = Parameters<OpenAI["responses"]["create"]>[1];

const isUnsupportedTemperatureError = (error: unknown) => {
  if (!(error instanceof Error)) {
    return false;
  }

  const message = error.message.toLowerCase();
  const status = (error as { status?: number }).status;

  return (
    message.includes("temperature") &&
    message.includes("not supported") &&
    (status === undefined || status === 400)
  );
};

export async function createResponseWithTemperatureFallback(
  openAi: OpenAI,
  payload: ResponsePayload,
  options?: ResponseOptions,
  logger?: Logger
): Promise<OpenAI.Responses.Response> {
  const hasTemperature =
    "temperature" in payload &&
    (payload as { temperature?: unknown }).temperature !== undefined;

  if (!hasTemperature) {
    return openAi.responses.create(
      payload,
      options
    ) as Promise<OpenAI.Responses.Response>;
  }

  try {
    return (await openAi.responses.create(
      payload,
      options
    )) as OpenAI.Responses.Response;
  } catch (error) {
    if (!isUnsupportedTemperatureError(error)) {
      throw error;
    }

    const { temperature, ...payloadWithoutTemperature } =
      payload as Record<string, unknown>;

    logger?.log(
      "warn",
      "openai",
      "Temperature not supported for model, retrying without it",
      { model: (payload as { model?: string }).model, temperature }
    );

    return openAi.responses.create(
      payloadWithoutTemperature as ResponsePayload,
      options
    ) as Promise<OpenAI.Responses.Response>;
  }
}
