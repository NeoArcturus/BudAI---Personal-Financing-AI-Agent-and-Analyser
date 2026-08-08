import { ChartConfiguration, ChartData } from "chart.js";

export interface Account {
  account_id: string;
  bank_name?: string;
  provider_name?: string;
  account_number?: string;
  sort_code?: string;
  currency?: string;
  balance?: number;
  status?: string;
  provider_id?: string;
  logo_url?: string;
  
  truelayer_account_id?: string;
}

export interface Transaction {
  transaction_id: string;
  date?: string;
  amount?: number;
  category?: string;
  merchant_name?: string;
  description?: string;
  status?: string;
  account_id?: string;
  bank_uuid?: string;
  user_uuid?: string;

  transaction_uuid?: string;
  timestamp?: string;
  Category?: string;
  Amount?: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

export interface LocalMessage extends ChatMessage {
  timestamp?: Date;
}

export type TabType =
  | "raw"
  | "categorized"
  | "categorized_doughnut"
  | "balance_forecast"
  | "expense_forecast"
  | "historical"
  | "historical_daily"
  | "historical_weekly"
  | "historical_monthly"
  | "cash_flow_mixed"
  | "health_radar";

export type NativeChartConfig = ChartConfiguration<
  "line" | "bar" | "doughnut" | "radar"
>;

export interface ChartDataPoint {
  Date?: string;
  Month?: string;
  Category?: string;
  Amount?: number;
  Total_Amount?: number;
  currency?: string;
  descriptions?: string[];
  [key: string]: any;
}

export interface BankChartData {
  bank_name: string;
  data: ChartDataPoint[];
}

export interface ChartConfig {
  type: string;
  data: BankChartData[];
}

export interface ToolParameters {
  bank_name_or_id?: string;
  account_ids?: string[];
  from_date?: string;
  to_date?: string;
  days?: number;
  plot_time_type?: string;
}

export interface NextAction {
  label: string;
  tool_target: string;
}

export type ExplanationContextType =
  | "CHART"
  | "TRANSACTION_LEDGER"
  | "MARKET_NEWS"
  | "ADVISORY"
  | string
  | null;

export type ExplanationPayload =
  | BankChartData[]
  | Transaction[]
  | Record<string, unknown>
  | unknown[]
  | ChartData<"line" | "bar" | "doughnut" | "radar">;

export interface ExplanationState {
  isOpen: boolean;
  isExplaining: boolean;
  contextType: ExplanationContextType;
  rawPayload: ExplanationPayload | null;
  aiExplanation: string;
  nextActions: NextAction[];
}

export interface CategorySummary {
  Category?: string;
  category?: string;
  Total_Amount?: number | string;
  amount?: number | string;
}

export interface ParsedCategory {
  name: string;
  value: number;
}
