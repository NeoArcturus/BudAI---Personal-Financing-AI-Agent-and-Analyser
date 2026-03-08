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
  | "balance_forecast"
  | "expense_forecast"
  | "historical";

export type NativeChartConfig = ChartConfiguration<"line" | "bar">;
