export interface Transaction {
  description?: string;
  Description?: string; // CSV mapping
  category?: string;
  Category?: string; // CSV mapping
  amount?: number;
  Amount?: number; // CSV mapping
  date?: string;
  Date?: string; // CSV mapping
  timestamp?: string; // TrueLayer mapping
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

export interface FinancialData {
  balance: string;
  transactions: Transaction[];
}

export interface Account {
  account_id: string;
  provider_name: string;
  account_number: string;
  sort_code: string;
  currency: string;
  balance: number;
}
