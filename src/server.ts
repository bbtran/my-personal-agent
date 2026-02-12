import { routeAgentRequest, type Schedule } from "agents";

import { AIChatAgent } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateId,
  stepCountIs,
  streamText,
  type StreamTextOnFinishCallback,
  type ToolSet
} from "ai";
import { createAiGateway } from "ai-gateway-provider";
import { createUnified } from "ai-gateway-provider/providers/unified";
import { env } from "cloudflare:workers";
import { executions, tools } from "./tools";
import {
  cleanupMessages,
  formatFlightResults,
  processToolCalls
} from "./utils";

const aiGateway = createAiGateway({
  accountId: "dbba1cf9c2d1adcbbc1e8ff9dbf91a6a",
  gateway: "ben-test-gateway",
  apiKey: env.CF_AIG_TOKEN
});

const unified = createUnified();

const model = aiGateway(
  unified("workers-ai/@cf/meta/llama-4-scout-17b-16e-instruct")
);

/**
 * Chat Agent implementation that handles real-time AI chat interactions
 */
export class ChatAgent extends AIChatAgent<Env> {
  /**
   * Handles incoming chat messages and manages the response stream
   */
  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<ToolSet>,
    options?: { abortSignal?: AbortSignal }
  ) {
    // Collect all tools
    const allTools = {
      ...tools
    };

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        // Clean up incomplete tool calls to prevent API errors
        const cleanedMessages = cleanupMessages(this.messages);

        // Process any pending tool calls from previous messages
        // This handles human-in-the-loop confirmations for tools
        const processedMessages = await processToolCalls({
          messages: cleanedMessages,
          dataStream: writer,
          tools: allTools,
          executions
        });

        const modelMessages = await convertToModelMessages(processedMessages);
        // console.log("=== DEBUG: Messages being sent to model ===");
        // console.log(JSON.stringify(modelMessages, null, 2));
        // console.log("=== END DEBUG ===");

        const result = streamText({
          system: `You are a helpful, friendly AI assistant. You can answer general questions, have conversations, help with analysis, writing, coding, math, and much more.

You also have access to tools for specific tasks:
- Get weather information for any city
- Get local time for any location  
- Schedule tasks to be executed later
- List and cancel scheduled tasks

Use tools when they are relevant to the user's request. For general questions and conversations, respond directly without using tools.

Current date and time: ${new Date().toISOString()}`,

          messages: modelMessages,
          model: model,
          tools: allTools,
          // Type boundary: streamText expects specific tool types, but base class uses ToolSet
          // This is safe because our tools satisfy ToolSet interface (verified by 'satisfies' in tools.ts)
          onFinish: onFinish as unknown as StreamTextOnFinishCallback<
            typeof allTools
          >,
          stopWhen: stepCountIs(10),
          toolChoice: "auto",
          abortSignal: options?.abortSignal
        });

        writer.merge(result.toUIMessageStream());
      }
    });

    return createUIMessageStreamResponse({ stream });
  }
  async executeTask(description: string, _task: Schedule<string>) {
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        parts: [
          {
            type: "text",
            text: `Running scheduled task: ${description}`
          }
        ],
        metadata: {
          createdAt: new Date()
        }
      }
    ]);
  }
}

/**
 * Flight Agent that connects to the Flights MCP server
 */
export class FlightAgent extends AIChatAgent<Env> {
  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<ToolSet>,
    options?: { abortSignal?: AbortSignal }
  ) {
    // Check if already connected, otherwise connect to the Flights MCP server
    const connected = await this.mcp.discoverIfConnected("flights-server");
    if (!connected) {
      const id = await this.mcp.registerServer("flights-server", {
        name: "flights-server",
        url: "https://flights-mcp.benjamin-tran25.workers.dev/mcp",
        callbackUrl: "https://flights-mcp.benjamin-tran25.workers.dev/callback"
      });
      await this.mcp.connectToServer(id);
    }

    // Get tools from MCP server
    const mcpTools = this.mcp.getAITools();

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        const cleanedMessages = cleanupMessages(this.messages);

        const processedMessages = await processToolCalls({
          messages: cleanedMessages,
          dataStream: writer,
          tools: { ...mcpTools },
          executions: {}
        });

        // Transform flight search results in messages to pre-formatted text
        const transformedMessages = processedMessages.map((msg) => {
          if (!msg.parts) return msg;
          const transformedParts = msg.parts.map((part) => {
            if (
              part.type.startsWith("tool-") &&
              "output" in part &&
              part.output &&
              typeof part.output === "object" &&
              "offers" in part.output
            ) {
              return {
                ...part,
                output: formatFlightResults(part.output)
              };
            }
            return part;
          });
          return { ...msg, parts: transformedParts };
        });

        const modelMessages = await convertToModelMessages(transformedMessages);

        const result = streamText({
          system: `You are a helpful flight booking assistant. You can help users search for flights, book flights, and manage their reservations.

Use the available tools to search for and book flights when the user requests.

The flight search tool returns pre-formatted results. Simply present these results to the user in a clear, readable format. Do not try to reparse or reformat the data.

Current date and time: ${new Date().toISOString()}`,

          messages: modelMessages,
          model: model,
          tools: mcpTools,
          onFinish: onFinish as unknown as StreamTextOnFinishCallback<
            typeof mcpTools
          >,
          stopWhen: stepCountIs(10),
          toolChoice: "auto",
          abortSignal: options?.abortSignal
        });

        writer.merge(result.toUIMessageStream());
      }
    });

    return createUIMessageStreamResponse({ stream });
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    // const url = new URL(request.url);
    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
