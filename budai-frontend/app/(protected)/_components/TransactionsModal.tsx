"use client";

import React from "react";
import { Receipt } from "lucide-react";
import { Transaction } from "@/types";
import TransactionFeed from "./TransactionFeed";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface TransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactions: Transaction[];
  bankName: string;
}

export default function TransactionModal({
  isOpen,
  onClose,
  transactions,
  bankName,
}: TransactionModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="bg-[#161B22] border-slate-700 max-w-4xl h-[85vh] p-0 flex flex-col shadow-2xl overflow-hidden gap-0">
        <DialogHeader className="p-6 border-b border-slate-800 bg-[#1c2128]">
          <DialogTitle className="text-lg font-bold text-white tracking-widest uppercase flex items-center gap-3">
            <Receipt className="text-[#00FFAA]" size={20} />
            {bankName} Ledger
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1 p-6">
          <TransactionFeed transactions={transactions} showCategory={true} />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
