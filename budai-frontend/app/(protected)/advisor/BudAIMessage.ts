import { UIMessage, tool, InferUITools, ToolSet } from "ai";
import z from "zod";

const budAIMetadataSchema = z.object({
  sessionId: z.string().uuid(),
  userUuid: z.string().uuid(),
  analysisDepth: z.enum(["standard", "intensive"]),
  timestamp: z.date(),
}).partial();

const budAIDataPartSchema = z.object({
  thinking_context: z.object({
    status: z.string(),
    worker_id: z.string(),
  }),
  global_refresh_signal: z.object({
    timeStamp: z.string(),
  }),
  htil_interrupt: z.object({
    id: z.string(),
    type: z.string(),
    payload: z.any(),
  }),
});

const budAITools = {
  render_ui_chart: tool({
    description: "Render a financial chart based on tool output",
    inputSchema: z.object({
      available_accounts: z
        .array(
          z.object({
            id: z.string(),
            name: z.string(),
          }),
        )
        .describe("List of verified accounts found in user profile"),
      reason: z
        .string()
        .describe("Contextual explanation for why selection is required"),
    }),
    outputSchema: z.object({
      selected_account_id: z
        .string()
        .describe("The UUID of the account chosen by the user"),
    }),
  }),

  execute_tool: tool({
    description: "Executes action based on the tool's backend operation",
    inputSchema: z.object({
      service_type: z.enum(["analyzer", "forecaster", "health", "market"]),
      query: z.string().describe("The specific data inquiry"),
    }),
    outputSchema: z.object({
      content: z.string(),
      chart_trigger: z
        .object({
          type: z.string(),
          cache_id: z.string(),
        })
        .optional(),
    }),
  }),
} satisfies ToolSet;

export type BudAITools = InferUITools<typeof budAITools>;
export type BudAIMetadata = z.infer<typeof budAIMetadataSchema>;
export type BudAIDataPart = z.infer<typeof budAIDataPartSchema>;

export type BudAIAnnotation =
  | { type: "thinking_context"; status: string; worker_id?: string }
  | { type: "global_refresh_signal"; chart_type?: string; timeStamp?: string }
  | { type: "htil_interrupt"; interrupts: Array<{ id: string; value: unknown }>; payload?: unknown };

export type BudAIMessage = UIMessage<BudAIMetadata, BudAIDataPart, BudAITools> & {
  annotations?: BudAIAnnotation[];
  usage?: {
    completionTokens?: number;
    promptTokens?: number;
    totalTokens?: number;
  };
};
