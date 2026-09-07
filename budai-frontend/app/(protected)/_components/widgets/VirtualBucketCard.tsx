"use client";

import React, { useEffect, useRef, useState } from "react";
import { Card, CardHeader, Meter, Label, Dropdown, Popover, Button } from "@heroui/react";
import { MoreVertical, Plus } from "lucide-react";
import { motion, useSpring, useTransform } from "framer-motion";
import { AnimatedNumber } from "../ui/AnimatedNumber";

export type BucketType = "DEFAULT" | "RECURRING" | "ACCUMULATING" | "TARGET_DATE" | "LIABILITY" | "UNALLOCATED";

interface VirtualBucketCardProps {
  type: BucketType;
  title: string;
  balance: number;
  currency?: string;
  sparklineData?: number[];
  progress?: number;
  targetAmount?: number;
  maxValue?: number;
  onEdit?: () => void;
  onDelete?: () => void;
  onAddFunds?: (amount: number) => void;
}

const QuickTransferPopover = ({ onTransfer }: { onTransfer: (amount: number) => void }) => {
  const [amount, setAmount] = useState("");
  const [isOpen, setIsOpen] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const num = parseFloat(amount);
    if (!isNaN(num) && num > 0) {
      onTransfer(num);
      setIsOpen(false);
      setAmount("");
    }
  };

  return (
    <Popover isOpen={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger>
        <button aria-label="Add funds" className="ml-4 p-1.5 rounded-full bg-white/5 hover:bg-white/10 text-foreground/50 hover:text-foreground transition-all cursor-pointer border border-white/10 outline-none focus:ring-1 focus:ring-primary/50">
          <Plus size={16} />
        </button>
      </Popover.Trigger>
      <Popover.Content placement="bottom" className="w-56 bg-black/95 backdrop-blur-3xl border border-white/10 rounded-xl shadow-2xl p-0">
        <Popover.Dialog className="p-4 w-full">
          <Popover.Heading className="text-[10px] font-black uppercase tracking-widest text-foreground/70 mb-3 w-full text-left">
            Add money
          </Popover.Heading>
          <form onSubmit={handleSubmit} className="flex flex-col gap-3 w-full">
            <div className="relative flex items-center bg-white/5 border border-white/10 focus-within:border-primary/50 transition-colors shadow-inner rounded-lg overflow-hidden">
              <span className="text-foreground/50 font-bold text-sm pl-3 absolute left-0">£</span>
              <input
                type="number"
                step="0.01"
                min="0"
                placeholder="0.00"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="w-full bg-transparent text-foreground font-bold text-sm py-2 pl-7 pr-3 outline-none focus:outline-none"
                autoFocus
              />
            </div>
            <Button type="submit" className="w-full bg-primary/20 text-primary hover:bg-primary/30 border border-primary/20 font-bold text-[10px] uppercase tracking-widest rounded-lg h-9 transition-colors">
              Transfer
            </Button>
          </form>
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  );
};

export const VirtualBucketCard = ({ type, title, balance, currency = "GBP", sparklineData = [], progress, targetAmount, maxValue, onEdit, onDelete, onAddFunds }: VirtualBucketCardProps) => {
  const getTheme = () => {
    switch (type) {
      case "LIABILITY": return { border: "border-red-500/30", bg: "bg-red-500/5", text: "text-red-500", highlight: "#EF4444" };
      case "ACCUMULATING": return { border: "border-green-500/30", bg: "bg-green-500/5", text: "text-green-500", highlight: "#10B981" };
      case "TARGET_DATE": return { border: "border-blue-500/30", bg: "bg-blue-500/5", text: "text-blue-500", highlight: "#3B82F6" };
      case "RECURRING": return { border: "border-yellow-500/30", bg: "bg-yellow-500/5", text: "text-yellow-500", highlight: "#EAB308" };
      case "DEFAULT":
      case "UNALLOCATED": return { border: "border-white/20", bg: "bg-white/5", text: "text-foreground/70", highlight: "#ffffff" };
      default: return { border: "border-primary/30", bg: "bg-primary/5", text: "text-primary", highlight: "#6366F1" };
    }
  };

  const theme = getTheme();

  const typeLabel = {
    TARGET_DATE: "Goal",
    ACCUMULATING: "Reserve",
    RECURRING: "Fixed Expense",
    LIABILITY: "Debt",
    DEFAULT: "Unallocated",
    UNALLOCATED: "Unallocated",
  }[type] || type;

  const displayBalance = Math.max(0, balance);

  return (
    <Card className="w-full h-full rounded-2xl flex flex-col relative min-h-[220px]">
      <CardHeader className="flex justify-between items-start w-full p-8 pb-4 z-10 shrink-0 relative">
        <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 pr-12">
          {title}
        </h3>
        <div className="flex items-center gap-2">
          <span className={`text-[8px] font-black uppercase tracking-widest px-2 py-1 rounded-md border ${theme.text} ${theme.bg} ${theme.border}`}>
            {typeLabel}
          </span>
          {type !== 'DEFAULT' && type !== 'UNALLOCATED' && (
            <div onPointerDown={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
              <Dropdown>
                <Dropdown.Trigger aria-label="Bucket options" className="cursor-pointer text-foreground/30 hover:text-foreground transition-colors p-1 rounded-full hover:bg-white/5 border border-transparent outline-none">
                  <MoreVertical size={14} />
                </Dropdown.Trigger>
                <Dropdown.Popover className="bg-black/90 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl min-w-[120px] p-1">
                  <Dropdown.Menu aria-label="Bucket options" onAction={(key) => {
                    const action = key.toString();
                    if (action === "edit" && onEdit) onEdit();
                    if (action === "delete" && onDelete) onDelete();
                  }}>
                    <Dropdown.Item id="edit" textValue="Edit" className="rounded-lg data-[hovered=true]:bg-primary/20 data-[hovered=true]:text-primary transition-all px-3 py-2 cursor-pointer outline-none mb-1">
                      <span className="font-bold text-xs tracking-wide">Edit</span>
                    </Dropdown.Item>
                    <Dropdown.Item id="delete" textValue="Delete" className="rounded-lg data-[hovered=true]:bg-[#f31260] data-[hovered=true]:text-white transition-all px-3 py-2 cursor-pointer outline-none text-[#f31260]">
                      <span className="font-bold text-xs tracking-wide">Delete</span>
                    </Dropdown.Item>
                  </Dropdown.Menu>
                </Dropdown.Popover>
              </Dropdown>
            </div>
          )}
        </div>
      </CardHeader>

      <div className="flex-1 w-full flex flex-col justify-center px-8 z-10 relative overflow-hidden">
        <div className="font-black text-3xl tracking-tighter text-foreground flex items-center">
          <AnimatedNumber value={displayBalance} />
          {type !== 'DEFAULT' && type !== 'UNALLOCATED' && onAddFunds && (
            <div onPointerDown={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
              <QuickTransferPopover onTransfer={onAddFunds} />
            </div>
          )}
        </div>
      </div>

      <div className="w-full px-8 pb-8 z-10 shrink-0 mt-auto">
        {targetAmount !== undefined && progress !== undefined ? (
          <Meter aria-label={`${title} target`} value={progress || 0} className="w-full flex flex-col gap-2 block">
            <div className="flex justify-between items-center w-full">
              <Label className="text-[9px] font-black uppercase tracking-widest text-foreground/50">
                Target: <AnimatedNumber value={targetAmount} />
              </Label>
              <Meter.Output className="text-[10px] font-mono font-bold tabular-nums text-foreground/80" />
            </div>
            <Meter.Track className="h-1.5 bg-black/40 rounded-full shadow-inner overflow-hidden">
              <Meter.Fill
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  backgroundColor: theme.highlight,
                  boxShadow: `0 0 12px ${theme.highlight}80`
                }}
              />
            </Meter.Track>
          </Meter>
        ) : (
          <Meter aria-label="Liquid Capital" value={Math.min(100, Math.max(0, (balance / (maxValue || 1)) * 100))} className="w-full flex flex-col gap-2 block">
            <div className="flex justify-between items-center w-full">
              <Label className="text-[9px] font-black uppercase tracking-widest text-foreground/30">
                {maxValue ? "Available of Total" : "Available money"}
              </Label>
              <Meter.Output className="text-[10px] font-mono font-bold tabular-nums text-foreground/50" />
            </div>
            <Meter.Track className="h-1.5 bg-black/40 rounded-full shadow-inner overflow-hidden">
              <Meter.Fill
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{ backgroundColor: theme.highlight, boxShadow: `0 0 12px ${theme.highlight}80` }}
              />
            </Meter.Track>
          </Meter>
        )}
      </div>
    </Card>
  );
};
