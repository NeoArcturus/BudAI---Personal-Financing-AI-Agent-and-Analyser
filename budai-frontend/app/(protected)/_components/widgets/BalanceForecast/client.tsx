"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { TrendingUp, TrendingDown, Clock, ShieldCheck, Target, ChevronDown } from "lucide-react";
import { Settings } from "lucide-react";
import { Card, Button, Skeleton, Dropdown, CloseButton, Badge, Label, Description } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "../../internal/ChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { BankChartData } from "@/types";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard, { FlipButton } from "../../internal/FlipCard";
import SimulationControlsModal, {
  SimulationOverrides,
} from "@/app/(protected)/_components/modals/SimulationControlsModal";

export default function BalanceForecastWidgetClient() {
  const { onRemove, instanceId } = React.useContext(WidgetContext);
  const { accounts } = useBudAI();
  const [isLoading, setIsLoading] = useState(false);
  const [wealthForecast, setWealthForecast] = useState<any>(null);

  const [localAccountId, setLocalAccountId] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [simulationOverrides, setSimulationOverrides] = useState<SimulationOverrides>({
    discipline_multiplier: 1.0,
    drift_adjustment: 0.0,
    macro_environment: "Stable",
    stress_test_active: false,
    days: 60,
  });

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  const fetchForecast = useCallback(async () => {
    if (!localAccountId) return;
    setIsLoading(true);
    try {
      const wealthRes = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          body: JSON.stringify({
            tool_name: "generate_financial_forecast",
            parameters: {
              bank_name_or_id: localAccountId,
              ...simulationOverrides,
            },
          }),
        },
        true
      );

      if (wealthRes.ok) {
        const result = await wealthRes.json() as any;
        setWealthForecast(result.data);
      }
    } catch (error) {
      console.error("Forecast Fetch Error:", error);
    } finally {
      setIsLoading(false);
    }
  }, [localAccountId, simulationOverrides]);

  useEffect(() => {
    fetchForecast();
  }, [fetchForecast]);

  const chartConfig = useMemo(() => {
    if (!wealthForecast) return null;
    return buildChartConfig(
      "balance_forecast",
      wealthForecast,
      { bank_name_or_id: localAccountId, days: simulationOverrides.days },
      `Projected Balance (${simulationOverrides.days} Days)`
    );
  }, [wealthForecast, localAccountId, simulationOverrides.days]);

  const stats = useMemo(() => {
    const series = (wealthForecast && typeof wealthForecast === 'object' && 'series' in wealthForecast) ? wealthForecast.series : (Array.isArray(wealthForecast) ? wealthForecast : [wealthForecast]);
    if (!series?.[0]?.data?.length) return { projected: 0, change: 0, current: 0 };
    const data = series[0].data;
    const current = Number(data[0]?.["Expected Balance"] || data[0]?.["Balance"] || 0);
    const projected = Number(data[data.length - 1]?.["Expected Balance"] || data[data.length - 1]?.["Balance"] || 0);
    const change = current !== 0 ? ((projected - current) / Math.abs(current)) * 100 : 0;
    return { projected, change, current };
  }, [wealthForecast]);

  const activeAccount = accounts.find(a => a.account_id === localAccountId);

  const frontContent = (
    <Card className="w-full h-full liquid-glass border-none rounded-xl flex flex-col relative overflow-hidden">
      <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
        <div className="flex justify-between items-start w-full">
          <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
            Balance Forecast
          </h3>
          <div className="flex items-center gap-1">
            <FlipButton />
            {onRemove && (
              <CloseButton
                onPress={onRemove}
                className="w-8 h-8 min-w-8 text-foreground/20 hover:text-foreground transition-all rounded-md"
              />
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-4 w-full pointer-events-auto">
          <div className="flex-1 min-w-40 max-w-50 ml-auto">
            <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mb-2 block pl-1">
              Selected Account
            </Label>
            <Dropdown>
              <Dropdown.Trigger className="h-12 min-h-12 w-full bg-white/5 hover:bg-white/10 border-[0.5px] border-white/10 text-[10px] text-foreground font-black uppercase tracking-widest rounded-xl px-4 flex items-center justify-between transition-all cursor-pointer outline-none focus:border-primary/50 shadow-inner">
                <span className="truncate pointer-events-none">
                  {activeAccount?.bank_name ? `${activeAccount.bank_name} (${activeAccount.currency || "GBP"})` : "Select Account"}
                </span>
                <ChevronDown
                  size={14}
                  className="text-foreground/30 shrink-0 pointer-events-none"
                />
              </Dropdown.Trigger>
              <Dropdown.Popover className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl w-64 z-50 p-2">
                <Dropdown.Menu
                  items={accounts.map(a => ({ ...a, id: a.account_id }))}
                  className="outline-none"
                  selectedKeys={new Set([localAccountId])}
                  onSelectionChange={(keys: any) => {
                    const val = Array.from(keys)[0] as string;
                    if (val) setLocalAccountId(val);
                  }}
                  selectionMode="single"
                >
                  {(acc: any) => (
                    <Dropdown.Item
                      key={acc.id}
                      id={acc.id}
                      textValue={acc.bank_name}
                      className="flex flex-col px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none transition-all"
                    >
                      <div className="w-full relative flex items-center justify-between">
                        <span className="text-[11px] font-black text-foreground uppercase tracking-tight pr-4 italic truncate">
                          {acc.bank_name} ({acc.currency || "GBP"})
                        </span>
                        {localAccountId === acc.id && (
                          <div className="bg-primary border-none w-1.5 h-1.5 min-w-0 p-0 relative transform-none rounded-full shrink-0 shadow-[0_0_10px_rgba(0,242,255,0.6)]" />
                        )}
                      </div>
                      <span className="text-foreground/20 text-[9px] font-mono tracking-widest mt-1.5 uppercase">
                        Account No: *{acc.account_number?.slice(-4)}
                      </span>
                    </Dropdown.Item>
                  )}
                </Dropdown.Menu>
              </Dropdown.Popover>
            </Dropdown>
          </div>
          <Button
            isIconOnly
            size="sm"
            variant="ghost"
            className="w-12 h-12 min-w-12 bg-white/5 text-foreground/60 hover:text-primary rounded-xl border-[0.5px] border-white/10 shadow-inner flex flex-row justify-center items-center"
            onPress={() => setIsModalOpen(true)}
          >
            <Settings size={16} />
          </Button>
        </div>
      </Card.Header>

      <Card.Content className="flex-1 w-full min-h-0 flex flex-col p-0 z-0 overflow-hidden relative">
        {isLoading ? (
          <Skeleton animationType="shimmer" className="w-full h-full rounded-lg bg-white/5" />
        ) : chartConfig ? (
          <CoreChartEngine config={chartConfig} />
        ) : (
          <div className="text-foreground/20 text-center opacity-40">
            <Clock size={24} className="mx-auto mb-2" />
            <p className="text-[8px] font-black uppercase tracking-[0.4em]">Awaiting Data...</p>
          </div>
        )}
      </Card.Content>

      <div className="mt-4 flex items-center justify-between shrink-0 border-t border-white/5 pt-4 px-8 pb-4">
        <div>
          <p className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em] mb-1">
            Projected End
          </p>
          <p className="text-xl font-normal text-primary tracking-tighter font-mono leading-none">
            £{stats.projected.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
          </p>
        </div>
        <div className={`text-right flex flex-col items-end`}>
          <p className={`text-[10px] flex items-center gap-1 font-black uppercase tracking-widest ${stats.change >= 0 ? "text-green-500" : "text-red-500"}`}>
            {stats.change >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {stats.change >= 0 ? "+" : ""}{stats.change.toFixed(1)}%
          </p>
          <p className="text-[8px] text-foreground/30 uppercase tracking-[0.2em] mt-1">Growth Rate</p>
        </div>
      </div>
      <SimulationControlsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onApply={setSimulationOverrides}
        initialValues={simulationOverrides}
      />
    </Card >
  );

  return (
    <WidgetFlipCard
      insight="Balance projections are mathematically modeled based on your historical cashflow velocity and current simulation parameters. Discuss this forecast to explore detailed 'what-if' scenarios."
      onDiscuss={() => {
        // Implement chat session opening if needed
      }}
    >
      {frontContent}
    </WidgetFlipCard>
  );
}

