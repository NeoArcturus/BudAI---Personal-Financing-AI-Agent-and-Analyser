import useSWR from 'swr';
import { apiFetch } from '@/lib/api';

const fetcher = async <T>(url: string): Promise<T> => {
  const res = await apiFetch(url, {}, true);
  if (!res.ok) {
    const error: any = new Error("Failed to fetch");
    error.status = res.status;
    throw error;
  }
  return res.json() as Promise<T>;
};

interface SubscriptionItem {
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

interface SubscriptionsData {
  subscriptions: SubscriptionItem[];
}

export function useSubscriptionsData(userUuid: string | null) {
  const { data, error, isLoading } = useSWR<SubscriptionsData>(
    userUuid ? `/api/analytics/subscriptions?user_uuid=${userUuid}` : null,
    fetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: true,
      onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
        // If the backend is locked by a heavy process, retry every 10s
        if (error.status === 423) {
          setTimeout(() => revalidate({ retryCount }), 10000);
          return;
        }
        
        // Never retry on 404.
        if (error.status === 404) return;

        // Only retry up to 3 times for other errors
        if (retryCount >= 3) return;

        // Default backoff for other errors
        setTimeout(() => revalidate({ retryCount }), 5000);
      }
    }
  );

  return {
    subscriptionsData: data,
    isLoading,
    isError: error,
    isLocked: error?.status === 423
  };
}
