// via https://github.com/vercel/ai/blob/main/examples/next-openai/app/api/use-chat-human-in-the-loop/utils.ts

import type {
  ToolCallOptions,
  ToolSet,
  UIMessage,
  UIMessageStreamWriter
} from "ai";
import { convertToModelMessages, isStaticToolUIPart } from "ai";
import { APPROVAL } from "./shared";

function isValidToolName<K extends PropertyKey, T extends object>(
  key: K,
  obj: T
): key is K & keyof T {
  return key in obj;
}

/**
 * Processes tool invocations where human input is required, executing tools when authorized.
 */
export async function processToolCalls<Tools extends ToolSet>({
  dataStream,
  messages,
  executions
}: {
  tools: Tools; // used for type inference
  dataStream: UIMessageStreamWriter;
  messages: UIMessage[];
  executions: Record<
    string,
    // biome-ignore lint/suspicious/noExplicitAny: needs a better type
    (args: any, context: ToolCallOptions) => Promise<unknown>
  >;
}): Promise<UIMessage[]> {
  // Process all messages, not just the last one
  const processedMessages = await Promise.all(
    messages.map(async (message) => {
      const parts = message.parts;
      if (!parts) return message;

      const processedParts = await Promise.all(
        parts.map(async (part) => {
          // Only process static tool UI parts (dynamic tools handled separately)
          if (!isStaticToolUIPart(part)) return part;

          const toolName = part.type.replace(
            "tool-",
            ""
          ) as keyof typeof executions;

          // Only process tools that require confirmation (are in executions object) and are in 'input-available' state
          if (!(toolName in executions) || part.state !== "output-available")
            return part;

          let result: unknown;

          if (part.output === APPROVAL.YES) {
            // User approved the tool execution
            if (!isValidToolName(toolName, executions)) {
              return part;
            }

            const toolInstance = executions[toolName];
            if (toolInstance) {
              result = await toolInstance(part.input, {
                messages: await convertToModelMessages(messages),
                toolCallId: part.toolCallId
              });
            } else {
              result = "Error: No execute function found on tool";
            }
          } else if (part.output === APPROVAL.NO) {
            result = "Error: User denied access to tool execution";
          } else {
            // If no approval input yet, leave the part as-is for user interaction
            return part;
          }

          // Forward updated tool result to the client.
          dataStream.write({
            type: "tool-output-available",
            toolCallId: part.toolCallId,
            output: result
          });

          // Return updated tool part with the actual result.
          return {
            ...part,
            output: result
          };
        })
      );

      return { ...message, parts: processedParts };
    })
  );

  return processedMessages;
}

/**
 * Format ISO 8601 datetime to readable time (e.g., "12:25 PM")
 */
function formatTime(isoString: string): string {
  const match = isoString.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!match) return isoString;
  const [, , , , hours, minutes] = match;
  const hour = parseInt(hours, 10);
  const ampm = hour >= 12 ? "PM" : "AM";
  const hour12 = hour % 12 || 12;
  return `${hour12}:${minutes} ${ampm}`;
}

/**
 * Format ISO 8601 duration to readable format (e.g., "1h 30m")
 */
function formatDuration(isoDuration: string): string {
  const match = isoDuration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
  if (!match) return isoDuration;
  const hours = match[1] ? `${match[1]}h` : "";
  const minutes = match[2] ? `${match[2]}m` : "";
  return `${hours} ${minutes}`.trim();
}

/**
 * Format flight search results into human-readable text
 */
export function formatFlightResults(data: unknown): string {
  if (typeof data !== "object" || data === null) return JSON.stringify(data);

  const flightData = data as {
    totalOffers?: number;
    offers?: Array<{
      offerId: string;
      price: { total: string; currency: string };
      airlines: string[];
      itineraries: Array<{
        duration: string;
        segments: Array<{
          departure: string;
          arrival: string;
          carrier: string;
          flightNumber: string;
          duration: string;
          stops: number;
        }>;
      }>;
      seatsAvailable: number;
    }>;
    dictionaries?: {
      carriers?: Record<string, string>;
    };
  };

  if (!flightData.offers || !Array.isArray(flightData.offers)) {
    return JSON.stringify(data);
  }

  const carriers = flightData.dictionaries?.carriers || {};

  const lines: string[] = [];
  lines.push(
    `Found ${flightData.totalOffers || flightData.offers.length} flights:\n`
  );

  for (const offer of flightData.offers) {
    const airlineCode = offer.airlines?.[0] || "Unknown";
    const airlineName = carriers[airlineCode] || airlineCode;
    const price = `${offer.price.currency === "EUR" ? "€" : "$"}${offer.price.total}`;

    for (const itinerary of offer.itineraries || []) {
      const totalDuration = formatDuration(itinerary.duration);
      const segments = itinerary.segments || [];

      if (segments.length === 1) {
        const seg = segments[0];
        const depTime = formatTime(
          seg.departure.split(" at ")[1] || seg.departure
        );
        const arrTime = formatTime(seg.arrival.split(" at ")[1] || seg.arrival);
        lines.push(`• ${airlineName} ${seg.flightNumber} - ${price}`);
        lines.push(
          `  Departs: ${depTime} → Arrives: ${arrTime} (${totalDuration}, nonstop)`
        );
        lines.push(`  Seats available: ${offer.seatsAvailable}`);
        lines.push("");
      } else {
        lines.push(`• ${airlineName} - ${price} (${segments.length} stops)`);
        for (const seg of segments) {
          const depTime = formatTime(
            seg.departure.split(" at ")[1] || seg.departure
          );
          const arrTime = formatTime(
            seg.arrival.split(" at ")[1] || seg.arrival
          );
          lines.push(
            `  ${seg.flightNumber}: ${depTime} → ${arrTime} (${formatDuration(seg.duration)})`
          );
        }
        lines.push(`  Total duration: ${totalDuration}`);
        lines.push(`  Seats available: ${offer.seatsAvailable}`);
        lines.push("");
      }
    }
  }

  return lines.join("\n");
}

/**
 * Clean up incomplete tool calls from messages before sending to API
 * Prevents API errors from interrupted or failed tool executions
 */
export function cleanupMessages(messages: UIMessage[]): UIMessage[] {
  return messages.filter((message) => {
    if (!message.parts) return true;

    // Filter out messages with incomplete tool calls
    const hasIncompleteToolCall = message.parts.some((part) => {
      if (!isStaticToolUIPart(part)) return false;
      // Remove tool calls that are still streaming or awaiting input without results
      return (
        part.state === "input-streaming" ||
        (part.state === "input-available" && !part.output && !part.errorText)
      );
    });

    return !hasIncompleteToolCall;
  });
}
