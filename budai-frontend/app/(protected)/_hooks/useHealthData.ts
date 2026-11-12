import useSWR from 'swr';
import { ChartData } from 'chart.js';

export interface HealthRecommendation {
  title: string;
  desc: string;
  type: string;
}

export interface HealthMetrics {
  overall_score: number;
  recommendations: HealthRecommendation[];
  metrics?: { Metric: string; Score: number }[];
}

export interface HealthDataResponse {
  radarData: any[] | null;
  metricsData: HealthMetrics | null;
}

const healthFetcher = async ([_key, _accountId]: [string, string]): Promise<HealthDataResponse> => {
  return {
    radarData: null,
    metricsData: {
      overall_score: 85,
      recommendations: [
        {
          title: "Maintain Reserve",
          desc: "Maintain liquid reserves to cover recurring commitments.",
          type: "savings",
        },
      ],
      metrics: [
        { Metric: "Liquidity", Score: 88 },
        { Metric: "Savings", Score: 84 },
        { Metric: "Debt", Score: 92 },
        { Metric: "Discretionary", Score: 80 },
      ],
    },
  };
};

export function useHealthData(localAccountId: string | null) {
  const { data, error, isLoading } = useSWR<HealthDataResponse>(
    localAccountId ? ['health-data', localAccountId] : null,
    healthFetcher,
    {
      revalidateOnFocus: false,
    }
  );

  return {
    healthData: data,
    isLoading,
    isError: error,
    isLocked: false,
  };
}
