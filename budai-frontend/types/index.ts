import { ChartConfiguration } from "chart.js";

export interface Account {
  account_id: string;
  truelayer_account_id?: string;
  provider_name?: string;
  bank_name?: string;
  account_number: string;
  sort_code: string;
  currency: string;
  balance: number;
  account_balance?: number;
  status?: string;
  provider_id?: string;
}

export interface Transaction {
  id?: string;
  transaction_uuid?: string;
  date?: string;
  timestamp?: string;
  amount?: number;
  Amount?: number;
  description?: string;
  Description?: string;
  category?: string;
  Category?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
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

export interface BankChartData {
  bank_name: string;
  data: Record<string, string | number>[];
}

export interface ChartConfig {
  type: string;
  data: BankChartData[];
}

export interface ToolParameters {
  bank_name_or_id: string;
  from_date?: string;
  to_date?: string;
  days?: number;
  plot_time_type?: string;
}
