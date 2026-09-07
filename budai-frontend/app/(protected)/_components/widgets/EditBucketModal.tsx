"use client";

import React, { useState, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { toast, Dropdown, DatePicker, DateField, Calendar, Modal } from "@heroui/react";
import { DateValue } from "@internationalized/date";
import { ChevronDown, Calendar as CalendarIcon } from "lucide-react";

interface EditBucketModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export const EditBucketModal = ({ isOpen, onOpenChange, bucket }: EditBucketModalProps & { bucket: any }) => {
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

  useEffect(() => {
    if (isOpen && bucket) {
      setBucketType(bucket.bucket_type || bucket.type || "");
      setName(bucket.name || bucket.title || "");
      setTargetAmount(bucket.target_amount ? bucket.target_amount.toString() : "");
      
      if (bucket.target_date) {
        try {
          const date = new Date(bucket.target_date);
          const { CalendarDate } = require("@internationalized/date");
          setTargetDate(new CalendarDate(date.getFullYear(), date.getMonth() + 1, date.getDate()));
        } catch(e) {}
      } else {
        setTargetDate(null);
      }
    }
  }, [isOpen, bucket]);

  const handleSave = async () => {
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
      } else {
        toast("Failed to update bucket.", { variant: "danger" } as any);
      }
    } catch (err) {
      toast("Network error while updating bucket.", { variant: "danger" } as any);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">

      <Modal.Dialog className="modal relative w-full max-w-md flex flex-col overflow-hidden pointer-events-auto bg-black/90 border border-white/5 shadow-2xl rounded-2xl">
        <div className="p-6 border-b border-white/10 flex justify-between items-center bg-white/5">
          <h2 className="text-foreground font-black uppercase tracking-widest text-sm">Create Money Bucket</h2>
          <button onClick={() => onOpenChange(false)} className="text-foreground/50 hover:text-white">&times;</button>
        </div>

        <div className="p-6 flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-xs font-bold uppercase tracking-widest text-foreground/70">Bucket Name</label>
            <input
              type="text"
              placeholder="e.g. Japan Trip 2027"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-foreground focus:outline-none focus:border-primary transition-all"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs font-bold uppercase tracking-widest text-foreground/70">Bucket Type</label>
            <Dropdown>
              <Dropdown.Trigger className="bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary/50 rounded-lg p-3 text-sm text-foreground focus:outline-none transition-all flex items-center justify-between cursor-pointer outline-none w-full">
                <div className="flex items-center w-full justify-between">
                  <span>{bucketTypes.find(t => t.key === bucketType)?.label || "Select Type"}</span>
                  <ChevronDown size={16} className="text-foreground/50" />
                </div>
              </Dropdown.Trigger>
              <Dropdown.Popover className="bg-black/90 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl p-2 min-w-[250px]">
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
                      className="rounded-lg data-[hovered=true]:bg-primary/20 data-[hovered=true]:text-primary transition-all px-3 py-2 cursor-pointer outline-none w-full"
                    >
                      <div className="flex flex-col gap-1 w-full">
                        <span className="font-bold text-sm tracking-wide">{t.label}</span>
                      </div>
                    </Dropdown.Item>
                  ))}
                </Dropdown.Menu>
              </Dropdown.Popover>
            </Dropdown>
          </div>

          {(bucketType === "TARGET_DATE" || bucketType === "ACCUMULATING" || bucketType === "LIABILITY") && (
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold uppercase tracking-widest text-foreground/70">Target Amount (£)</label>
              <input
                type="number"
                placeholder="0.00"
                value={targetAmount}
                onChange={(e) => setTargetAmount(e.target.value)}
                className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-foreground focus:outline-none focus:border-primary transition-all"
              />
            </div>
          )}

          {bucketType === "TARGET_DATE" && (
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold uppercase tracking-widest text-foreground/70">Target Date</label>
              <DatePicker
                className="w-full"
                value={targetDate}
                onChange={setTargetDate}
              >
                <DateField.Group
                  fullWidth
                  className="bg-white/5 border border-white/10 rounded-lg p-3 h-auto flex items-center transition-all focus-within:border-primary/50"
                >
                  <DateField.Input className="flex-1  text-foreground text-sm font-sans ">
                    {(segment) => (
                      <DateField.Segment
                        segment={segment}
                        className="focus:bg-primary/20 rounded-md px-1 outline-none"
                      />
                    )}
                  </DateField.Input>
                  <DateField.Suffix className="ml-2 flex items-center">
                    <DatePicker.Trigger className="text-foreground/50 hover:text-primary cursor-pointer transition-colors outline-none">
                      <CalendarIcon size={16} />
                    </DatePicker.Trigger>
                  </DateField.Suffix>
                </DateField.Group>
                <DatePicker.Popover className="popover min-w-max p-6">
                  <Calendar aria-label="Target date" className="w-full min-w-[250px]">
                    <Calendar.Header className="flex items-center gap-3 mb-6">
                      <Calendar.YearPickerTrigger className="flex items-center gap-2 mr-auto cursor-pointer hover:opacity-70 transition-opacity outline-none">
                        <Calendar.YearPickerTriggerHeading className="text-sm font-black uppercase tracking-widest text-primary italic" />
                        <Calendar.YearPickerTriggerIndicator className="text-foreground/30 w-4 h-4" />
                      </Calendar.YearPickerTrigger>
                      <Calendar.NavButton
                        slot="previous"
                        className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors outline-none"
                      />
                      <Calendar.NavButton
                        slot="next"
                        className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors outline-none"
                      />
                    </Calendar.Header>
                    <Calendar.Grid className="w-full border-collapse">
                      <Calendar.GridHeader>
                        {(day) => (
                          <Calendar.HeaderCell className="text-[9px] font-black text-foreground/20 pb-4 text-center uppercase tracking-widest">
                            {day}
                          </Calendar.HeaderCell>
                        )}
                      </Calendar.GridHeader>
                      <Calendar.GridBody>
                        {(date) => (
                          <Calendar.Cell
                            date={date}
                            className="w-8 h-8 flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                          />
                        )}
                      </Calendar.GridBody>
                    </Calendar.Grid>
                    <Calendar.YearPickerGrid>
                      <Calendar.YearPickerGridBody>
                        {({year}) => (
                          <Calendar.YearPickerCell
                            year={year}
                            className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                          />
                        )}
                      </Calendar.YearPickerGridBody>
                    </Calendar.YearPickerGrid>
                  </Calendar>
                </DatePicker.Popover>
              </DatePicker>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-white/10 flex justify-end gap-3 bg-white/5">
          <button
            onClick={() => onOpenChange(false)}
            className="px-6 py-2 rounded-lg font-bold text-xs uppercase tracking-widest text-foreground/70 hover:bg-white/10 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSubmitting}
            className="px-6 py-2 rounded-lg font-bold text-xs uppercase tracking-widest bg-primary text-black hover:bg-primary/80 transition-all disabled:opacity-50"
          >
            {isSubmitting ? "Saving..." : "Save Changes"}
          </button>
        </div>
    </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  );
};
