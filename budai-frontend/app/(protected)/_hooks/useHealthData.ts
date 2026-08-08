import useSWR from 'swr';
import { apiFetch } from '@/lib/api';
import { ChartData } from 'chart.js';

interface HealthRecommendation {
  title: string;
  desc: string;
  type: string;
}

interface HealthMetrics {
  overall_score: number;
  recommendations: HealthRecommendation[];
  metrics?: { Metric: string; Score: number }[];
}

interface HealthDataResponse {
  radarData: any[] | null;
  metricsData: HealthMetrics | null;
}

const healthFetcher = async ([_key, localAccountId]: [string, string]): Promise<HealthDataResponse> => {
  const metricsRes = await apiFetch(
    "/api/media/execute",
    {
      method: "POST",
      body: JSON.stringify({
        tool_name: "get_financial_health_metrics",
        parameters: { user_uuid: "CURRENT_USER" },
      }),
    },
    true,
  );

  if (!metricsRes.ok) {
    const error: any = new Error("Failed to fetch metrics");
    error.status = metricsRes.status;
    throw error;
  }

  const metricsResult = await metricsRes.json() as any;
  let metricsData = null;
  let radarData = null;

  if (metricsResult && metricsResult.data) {
    try {
      metricsData = typeof metricsResult.data === "string" ? JSON.parse(metricsResult.data) : metricsResult.data;
      if (metricsData) {
        metricsData.overall_score = Number(metricsData.overall_score) || 0;
        
        if (Array.isArray(metricsData.recommendations)) {
          metricsData.recommendations = metricsData.recommendations.map((r: any) => ({
            ...r,
            desc: typeof r.desc === "string" ? r.desc.replace(/nan/gi, "0") : r.desc
          }));
        }

        if (Array.isArray(metricsData.metrics)) {
          metricsData.metrics = metricsData.metrics.map((m: any) => ({
            ...m,
            Score: Number(m.Score) || 0
          }));
          radarData = [{ bank_name: "Overall Health", data: metricsData.metrics }];
        }
      }
    } catch (e) {
      console.error("Failed to parse metrics data", e);
    }
  }

  return {
    radarData,
    metricsData
  };
};

export function useHealthData(localAccountId: string | null) {
  const { data, error, isLoading } = useSWR<HealthDataResponse>(
    localAccountId ? ['health-data', localAccountId] : null,
    healthFetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: true,
      onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
        if (error.status === 423) {
          setTimeout(() => revalidate({ retryCount }), 10000);
          return;
        }
        if (error.status === 404) return;
        if (retryCount >= 3) return;
        setTimeout(() => revalidate({ retryCount }), 5000);
      }
    }
  );

  return {
    healthData: data,
    isLoading,
    isError: error,
    isLocked: error?.status === 423
  };
}
