export interface SubscriptionItem {
  merchant_name: string;
  bank_name?: string;
  account_number?: string;
  sort_code?: string;
  expected_amount: number;
  last_payment_amount?: number;
  last_payment_date?: string;
  predicted_frequency: string;
  next_expected_date?: string;
  is_price_hike: boolean;
  status?: "303-200" | "303-410";
}

export interface SubscriptionsData {
  subscriptions: SubscriptionItem[];
}

export function useSubscriptionsData(_userUuid: string | null) {
  return {
    subscriptionsData: { subscriptions: [] } as SubscriptionsData,
    isLoading: false,
    isError: null,
    isLocked: false,
  };
}
