"use client";

import React, { useState } from "react";
import { Bell, BrainCircuit } from "lucide-react";
import { Button, SearchField, Tooltip } from "@heroui/react";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

interface TransactionsClientProps {
  ledgerTable: React.ReactNode;
}

export default function TransactionsClient({
  ledgerTable,
}: TransactionsClientProps) {
  const queryClient = useQueryClient();
  const [isRetraining, setIsRetraining] = useState(false);

  const handleRetrainAll = async () => {
    setIsRetraining(true);
    try {
      const res = await apiFetch(
        "/api/categorizer/retrain",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ force: true }),
        },
        true,
      );

      if (res.ok) {
        const data = await res.json();
        const taskId = data.task_id;

        if (taskId) {
          const pollInterval = setInterval(async () => {
            try {
              const statusRes = await apiFetch(
                `/api/categorizer/task-status/${taskId}`,
                {},
                true,
              );
              if (statusRes.ok) {
                const statusData = await statusRes.json();
                if (
                  statusData.status === "completed" ||
                  statusData.status === "failed"
                ) {
                  clearInterval(pollInterval);
                  queryClient.invalidateQueries({ queryKey: ["transactions"] });
                  setIsRetraining(false);
                }
              }
            } catch (e) {
              console.log(e);
              clearInterval(pollInterval);
              setIsRetraining(false);
            }
          }, 2000);
        } else {
          queryClient.invalidateQueries({ queryKey: ["transactions"] });
          setIsRetraining(false);
        }
      } else {
        setIsRetraining(false);
      }
    } catch (e) {
      console.error(e);
      setIsRetraining(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col pt-10 px-10 h-full">
      <div className="flex items-center justify-between mb-10 shrink-0">
        <div>
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Ledger <span className="font-normal not-italic">Transactions</span>
          </h2>
          <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] mt-1.5">
            Your transaction history
          </p>
        </div>
        <div className="flex items-center gap-6">
          <SearchField>
            <SearchField.Group className="flex flex-row border-[0.5px] rounded-xl py-2 px-4 justify-center items-center bg-white/5 border-white/10 hover:border-primary/50 transition-all shadow-inner">
              <SearchField.SearchIcon className="text-foreground/30" />
              <SearchField.Input
                placeholder="Search intelligence..."
                className="w-75 border-none outline-none ring-0 focus:outline-none focus:ring-0 px-3 text-[11px] font-medium tracking-wide placeholder:text-foreground/20"
              />
            </SearchField.Group>
          </SearchField>
          <Tooltip>
            <Tooltip.Content
              placement="top"
              className="bg-black border border-white/10 text-xs"
            >
              Retrain AI Model
            </Tooltip.Content>
            <Button
              isIconOnly
              variant="secondary"
              isDisabled={isRetraining}
              onPress={handleRetrainAll}
              className="w-11 h-11 min-w-11 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 text-primary shadow-md relative cursor-pointer flex justify-center items-center hover:bg-white/10"
            >
              <BrainCircuit size={18} />
            </Button>
          </Tooltip>

          <Button
            isIconOnly
            variant="secondary"
            className="w-11 h-11 min-w-11 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 text-foreground shadow-md relative cursor-pointer flex items-center justify-center hover:bg-white/10"
          >
            <Bell size={18} />
            <span className="absolute top-3 right-3 w-2.5 h-2.5 bg-destructive rounded-full border-2 border-black shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden pb-8 relative">
        <div className="h-full w-full liquid-glass rounded-3xl overflow-hidden scrollbar-hide shadow-2xl">
          {ledgerTable}
        </div>
      </div>
    </div>
  );
}
