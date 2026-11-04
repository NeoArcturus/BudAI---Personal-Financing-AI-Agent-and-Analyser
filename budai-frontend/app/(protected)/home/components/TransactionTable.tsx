"use client";

import React from "react";
import { Card, Table, Chip } from "@heroui/react";
import { AlertTriangle } from "lucide-react";

export interface TransactionItem {
  id: string;
  date: string;
  display_date: string;
  merchant: string;
  category: string;
  pool: string;
  bucket_type: string;
  amount: number;
  is_outflow: boolean;
  is_anomaly: boolean;
  status: "Verified" | "Reconciled" | "Action Required" | string;
}

interface TransactionTableProps {
  transactions: TransactionItem[];
}

export function TransactionTable({ transactions = [] }: TransactionTableProps) {
  const flaggedCount = transactions.filter(
    (t) => t.is_anomaly || t.status === "Action Required"
  ).length;

  return (
    <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3 w-full">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Monitored Transactions
          </h2>
          <p className="text-xs text-slate-500">
            Autonomous tagging, allocation & anomaly screening
          </p>
        </div>
        {flaggedCount > 0 && (
          <Chip size="sm" variant="soft" color="warning" className="font-mono text-[11px]">
            {flaggedCount} Flagged for Review
          </Chip>
        )}
      </div>

      {/* HeroUI Table */}
      <Table>
        <Table.ScrollContainer>
          <Table.Content aria-label="Monitored Transactions Table" className="w-full text-left text-xs">
            <Table.Header>
              <Table.Column isRowHeader className="py-2 text-slate-400 font-medium">Date</Table.Column>
              <Table.Column className="py-2 text-slate-400 font-medium">Merchant</Table.Column>
              <Table.Column className="py-2 text-slate-400 font-medium">Category</Table.Column>
              <Table.Column className="py-2 text-slate-400 font-medium">Pool</Table.Column>
              <Table.Column className="py-2 text-slate-400 font-medium text-right">Amount</Table.Column>
              <Table.Column className="py-2 text-slate-400 font-medium text-right">Agent Status</Table.Column>
            </Table.Header>
            <Table.Body>
              {transactions.map((tx) => {
                const isActionRequired =
                  tx.status === "Action Required" || tx.is_anomaly;

                if (isActionRequired) {
                  return (
                    <Table.Row
                      key={tx.id}
                      className="bg-amber-950/20 hover:bg-amber-950/30 transition border-l-2 border-amber-500"
                    >
                      <Table.Cell className="py-2.5 pl-2 text-slate-400 font-mono text-[11px]">
                        {tx.display_date}
                      </Table.Cell>
                      <Table.Cell className="py-2.5 font-medium text-amber-200">
                        <div className="flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                          <span className="truncate">{tx.merchant}</span>
                        </div>
                      </Table.Cell>
                      <Table.Cell className="py-2.5 text-amber-300/80 italic font-mono text-[11px]">
                        {tx.category}
                      </Table.Cell>
                      <Table.Cell className="py-2.5">
                        <Chip size="sm" variant="soft" color="warning" className="font-mono text-[11px]">
                          {tx.pool}
                        </Chip>
                      </Table.Cell>
                      <Table.Cell className="py-2.5 text-right font-mono font-bold text-amber-200 tabular-nums">
                        £{tx.amount.toFixed(2)}
                      </Table.Cell>
                      <Table.Cell className="py-2.5 text-right">
                        <Chip size="sm" variant="soft" color="warning" className="font-mono text-[11px]">
                          Action Required
                        </Chip>
                      </Table.Cell>
                    </Table.Row>
                  );
                }

                return (
                  <Table.Row key={tx.id} className="hover:bg-slate-800/30 transition border-b border-slate-800/40">
                    <Table.Cell className="py-2.5 text-slate-400 font-mono text-[11px]">
                      {tx.display_date}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 font-medium text-white">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 shrink-0" />
                        <span className="truncate">{tx.merchant}</span>
                      </div>
                    </Table.Cell>
                    <Table.Cell className="py-2.5 text-slate-300">{tx.category}</Table.Cell>
                    <Table.Cell className="py-2.5">
                      <Chip size="sm" variant="soft" color="default" className="font-mono text-[11px]">
                        {tx.pool}
                      </Chip>
                    </Table.Cell>
                    <Table.Cell className="py-2.5 text-right font-mono font-medium text-white tabular-nums">
                      £{tx.amount.toFixed(2)}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 text-right">
                      <Chip
                        size="sm"
                        variant="soft"
                        color={tx.status === "Reconciled" ? "accent" : "success"}
                        className="font-mono text-[11px]"
                      >
                        {tx.status}
                      </Chip>
                    </Table.Cell>
                  </Table.Row>
                );
              })}
            </Table.Body>
          </Table.Content>
        </Table.ScrollContainer>
      </Table>
    </Card>
  );
}
