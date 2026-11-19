"use client";

import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { toast, Dropdown, DatePicker, DateField, Calendar, Modal, Button } from "@heroui/react";
import { DateValue } from "@internationalized/date";
import { ChevronDown, Calendar as CalendarIcon } from "lucide-react";

interface CreateBucketModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export const CreateBucketModal = ({ isOpen, onOpenChange }: CreateBucketModalProps) => {
  const queryClient = useQueryClient();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [name, setName] = useState("");
  const [bucketType, setBucketType] = useState<string>("TARGET_DATE");
  const [targetAmount, setTargetAmount] = useState("");
  const [targetDate, setTargetDate] = useState<DateValue | null>(null);

  const bucketTypes = [
    { key: "TARGET_DATE", label: "Goal" },
    { key: "RECURRING", label: "Fixed Expense" },
    { key: "ACCUMULATING", label: "Reserve" },
    { key: "LIABILITY", label: "Debt" },
  ];

  const handleCreate = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!name || !bucketType) {
      toast("Please fill out the bucket name and type.", { variant: "danger" } as any);
      return;
    }

    setIsSubmitting(true);
    try {
      const payload: any = {
        name,
        bucket_type: bucketType,
        priority_index: 1.0,
      };

      if (targetAmount) {
        payload.target_amount = parseFloat(targetAmount);
      }

      if (targetDate) {
        payload.target_date = new Date(`${targetDate.year}-${String(targetDate.month).padStart(2, "0")}-${String(targetDate.day).padStart(2, "0")}`).toISOString();
      }

      const res = await apiFetch("/api/buckets/", {
        method: "POST",
        body: payload as any,
      }, true);

      if (res.ok) {
        toast("Bucket created successfully.");
        queryClient.invalidateQueries({ queryKey: ["buckets"] });
        setName("");
        setBucketType("TARGET_DATE");
        setTargetAmount("");
        setTargetDate(null);
        onOpenChange(false);
      }
    } catch (err: any) {
      toast(err?.message || "Failed to create bucket.", { variant: "danger" } as any);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">
        <Modal.Dialog className="modal relative w-full max-w-md flex flex-col overflow-hidden pointer-events-auto bg-popover border border-border shadow-2xl rounded-2xl p-6">
          <div className="flex items-center justify-between border-b border-border/50 pb-4 mb-4">
            <div>
              <h2 className="text-base font-bold text-foreground">Create Money Bucket</h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Define a new virtual financial allocation
              </p>
            </div>
          </div>

          <form onSubmit={handleCreate} className="flex flex-col gap-4">
            <div>
              <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                Bucket Name
              </label>
              <div className="relative flex items-center bg-foreground/5 border border-border rounded-xl focus-within:border-primary/50 transition-colors">
                <input
                  type="text"
                  placeholder="e.g. Japan Trip 2027"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-transparent text-foreground font-bold text-sm py-2.5 px-3 outline-none"
                  autoFocus
                />
              </div>
            </div>

            <div>
              <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                Bucket Type
              </label>
              <Dropdown>
                <Button
                  variant="secondary"
                  className="w-full flex items-center justify-between px-3 py-2 h-auto rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border text-xs font-semibold transition-all text-left shadow-none"
                >
                  <div className="flex flex-col truncate">
                    <span className="truncate text-foreground">
                      {bucketTypes.find((t) => t.key === bucketType)?.label || "Select Type"}
                    </span>
                  </div>
                  <ChevronDown size={14} className="text-muted-foreground shrink-0" />
                </Button>
                <Dropdown.Popover className="bg-popover border border-border shadow-xl rounded-xl w-56 p-1.5 z-50">
                  <Dropdown.Menu
                    selectedKeys={new Set([bucketType])}
                    onAction={(key) => setBucketType(key as string)}
                    selectionMode="single"
                    className="flex flex-col gap-1 w-full"
                  >
                    {bucketTypes.map((t) => (
                      <Dropdown.Item
                        key={t.key}
                        id={t.key}
                        textValue={t.label}
                        className="px-3 py-2 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer"
                      >
                        <span className="font-medium text-foreground">{t.label}</span>
                      </Dropdown.Item>
                    ))}
                  </Dropdown.Menu>
                </Dropdown.Popover>
              </Dropdown>
            </div>

            {(bucketType === "TARGET_DATE" ||
              bucketType === "ACCUMULATING" ||
              bucketType === "LIABILITY") && (
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                  Target Amount (£)
                </label>
                <div className="relative flex items-center bg-foreground/5 border border-border rounded-xl focus-within:border-primary/50 transition-colors">
                  <span className="text-foreground/50 font-bold text-sm pl-3 absolute left-0">
                    £
                  </span>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    placeholder="0.00"
                    value={targetAmount}
                    onChange={(e) => setTargetAmount(e.target.value)}
                    className="w-full bg-transparent text-foreground font-bold text-sm py-2.5 pl-8 pr-3 outline-none"
                  />
                </div>
              </div>
            )}

            {bucketType === "TARGET_DATE" && (
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                  Target Date
                </label>
                <DatePicker
                  className="w-full"
                  value={targetDate}
                  onChange={setTargetDate}
                >
                  <DateField.Group
                    fullWidth
                    className="relative flex items-center bg-foreground/5 border border-border rounded-xl px-3 py-2 h-auto transition-colors focus-within:border-primary/50"
                  >
                    <DateField.Input className="flex-1 text-foreground text-sm font-sans">
                      {(segment) => (
                        <DateField.Segment
                          segment={segment}
                          className="focus:bg-primary/20 rounded px-1 outline-none"
                        />
                      )}
                    </DateField.Input>
                    <DateField.Suffix className="ml-2 flex items-center">
                      <DatePicker.Trigger className="text-muted-foreground hover:text-foreground cursor-pointer transition-colors outline-none">
                        <CalendarIcon size={16} />
                      </DatePicker.Trigger>
                    </DateField.Suffix>
                  </DateField.Group>
                  <DatePicker.Popover className="bg-popover border border-border shadow-2xl rounded-2xl p-4 min-w-max z-50">
                    <Calendar aria-label="Target date" className="w-full min-w-[250px]">
                      <Calendar.Header className="flex items-center gap-3 mb-4">
                        <Calendar.YearPickerTrigger className="flex items-center gap-2 mr-auto cursor-pointer hover:opacity-70 transition-opacity outline-none">
                          <Calendar.YearPickerTriggerHeading className="text-xs font-bold text-foreground" />
                          <Calendar.YearPickerTriggerIndicator className="text-muted-foreground w-3.5 h-3.5" />
                        </Calendar.YearPickerTrigger>
                        <Calendar.NavButton
                          slot="previous"
                          className="text-muted-foreground hover:text-foreground w-7 h-7 flex items-center justify-center rounded-lg hover:bg-foreground/5 cursor-pointer transition-colors outline-none"
                        />
                        <Calendar.NavButton
                          slot="next"
                          className="text-muted-foreground hover:text-foreground w-7 h-7 flex items-center justify-center rounded-lg hover:bg-foreground/5 cursor-pointer transition-colors outline-none"
                        />
                      </Calendar.Header>
                      <Calendar.Grid className="w-full border-collapse">
                        <Calendar.GridHeader>
                          {(day) => (
                            <Calendar.HeaderCell className="text-[10px] font-semibold text-muted-foreground pb-2 text-center uppercase tracking-wider">
                              {day}
                            </Calendar.HeaderCell>
                          )}
                        </Calendar.GridHeader>
                        <Calendar.GridBody>
                          {(date) => (
                            <Calendar.Cell
                              date={date}
                              className="w-8 h-8 flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-foreground/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                            />
                          )}
                        </Calendar.GridBody>
                      </Calendar.Grid>
                      <Calendar.YearPickerGrid>
                        <Calendar.YearPickerGridBody>
                          {({year}) => (
                            <Calendar.YearPickerCell
                              year={year}
                              className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-foreground/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                            />
                          )}
                        </Calendar.YearPickerGridBody>
                      </Calendar.YearPickerGrid>
                    </Calendar>
                  </DatePicker.Popover>
              </DatePicker>
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-2 border-t border-border/50">
            <Button
              variant="ghost"
              onPress={() => onOpenChange(false)}
              className="text-xs font-semibold px-4 py-2 rounded-xl"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              isDisabled={isSubmitting}
              isPending={isSubmitting}
              className="bg-primary text-primary-foreground text-xs font-bold px-5 py-2 rounded-xl shadow-sm hover:opacity-90 transition-opacity"
            >
              {isSubmitting ? "Creating..." : "Create Bucket"}
            </Button>
          </div>
        </form>
      </Modal.Dialog>
    </Modal.Container>
  </Modal.Backdrop>
);
};
