"use client";

import React, { useState, useEffect } from "react";
import {
  Modal,
  Button,
  Slider,
  Select,
  ListBox,
  Switch,
  Label,
  Description,
} from "@heroui/react";
import { Sparkles, Zap, ShieldAlert, Clock, ChevronDown } from "lucide-react";

export interface SimulationOverrides {
  discipline_multiplier: number;
  drift_adjustment: number;
  macro_environment: string;
  stress_test_active: boolean;
  days: number;
}

interface SimulationControlsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApply: (overrides: SimulationOverrides) => void;
  initialValues: SimulationOverrides;
}

export default function SimulationControlsModal({
  isOpen,
  onClose,
  onApply,
  initialValues,
}: SimulationControlsModalProps) {
  const [draft, setDraft] = useState<SimulationOverrides>(initialValues);

  useEffect(() => {
    if (isOpen) {
      setDraft(initialValues);
    }
  }, [isOpen, initialValues]);

  const handleApply = () => {
    onApply(draft);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Modal.Backdrop className="fixed inset-0 z-100 bg-black/60 backdrop-blur-sm">
        <Modal.Container className="fixed inset-0 z-101 flex items-center justify-center p-4">
          <Modal.Dialog className="relative max-w-md w-full pointer-events-auto obsidian-glass rounded-3xl border border-neon-cyan/20 shadow-[0_0_50px_rgba(0,229,255,0.15)] font-geist">
          <Modal.Header className="flex items-center gap-3 p-6 border-b border-white/5">
            <div className="w-10 h-10 rounded-xl bg-neon-cyan/10 flex items-center justify-center text-neon-cyan shadow-[inset_0_0_15px_rgba(0,229,255,0.1)]">
              <Zap size={20} />
            </div>
            <div>
              <Modal.Heading className="text-white font-bold text-xl tracking-tight">
                Simulation Controls
              </Modal.Heading>
              <p className="text-[#8B8E98] text-xs font-medium">
                Adjust hybrid LSTM-Bates engine constraints
              </p>
            </div>
          </Modal.Header>

          <Modal.Body className="p-6 space-y-8">
            {/* Spending Discipline */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-white text-sm font-bold uppercase tracking-wider">
                  Spending Discipline
                </Label>
                <span className="text-neon-cyan text-xs font-mono font-bold px-2 py-0.5 rounded-md bg-neon-cyan/10 border border-neon-cyan/20">
                  {draft.discipline_multiplier < 0.8
                    ? "Strict"
                    : draft.discipline_multiplier > 1.2
                      ? "Erratic"
                      : "Balanced"}
                </span>
              </div>
              <Slider
                minValue={0.5}
                maxValue={1.5}
                step={0.1}
                value={draft.discipline_multiplier}
                onChange={(v) =>
                  setDraft({ ...draft, discipline_multiplier: v as number })
                }
                className="w-full"
              >
                <Slider.Track className="bg-white/5 h-1.5 rounded-full">
                  <Slider.Fill className="bg-neon-cyan shadow-[0_0_10px_rgba(0,229,255,0.5)]" />
                  <Slider.Thumb className="w-5 h-5 bg-white border-2 border-neon-cyan shadow-lg" />
                </Slider.Track>
              </Slider>
              <Description className="text-[10px] text-[#5E6272] leading-tight">
                Controls volatility (theta/xi) multipliers. Lower values
                simulate more predictable spending.
              </Description>
            </div>

            {/* Target Savings Drift */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-white text-sm font-bold uppercase tracking-wider">
                  Drift Adjustment
                </Label>
                <span className="text-[#00E5FF] text-xs font-mono font-bold">
                  {(draft.drift_adjustment * 100).toFixed(1)}% / mo
                </span>
              </div>
              <Slider
                minValue={-0.05}
                maxValue={0.05}
                step={0.005}
                value={draft.drift_adjustment}
                onChange={(v) =>
                  setDraft({ ...draft, drift_adjustment: v as number })
                }
                className="w-full"
              >
                <Slider.Track className="bg-white/5 h-1.5 rounded-full">
                  <Slider.Fill className="bg-[#00E5FF] shadow-[0_0_10px_rgba(0,229,255,0.5)]" />
                  <Slider.Thumb className="w-5 h-5 bg-white border-2 border-[#00E5FF] shadow-lg" />
                </Slider.Track>
              </Slider>
              <Description className="text-[10px] text-[#5E6272] leading-tight">
                Directly offsets the baseline growth rate (mu). Positive for
                aggressive savings goals.
              </Description>
            </div>

            {/* Economic Environment */}
            <div className="space-y-3">
              <Label className="text-white text-sm font-bold uppercase tracking-wider">
                Economic Environment
              </Label>
              <Select
                value={draft.macro_environment}
                onChange={(v) =>
                  setDraft({ ...draft, macro_environment: v as string })
                }
                className="w-full"
              >
                <Select.Trigger className="bg-white/5 border-white/10 hover:border-white/20 h-12 rounded-xl px-4 flex justify-between items-center w-full transition-colors">
                  <Select.Value className="text-white text-sm font-medium" />
                  <ChevronDown size={16} className="text-[#5E6272]" />
                </Select.Trigger>
                <Select.Popover className="bg-[#121212] border border-white/10 rounded-xl shadow-2xl z-100">
                  <ListBox className="p-1">
                    {[
                      {
                        id: "Stable",
                        label: "Stable",
                        desc: "Standard historical parameters",
                      },
                      {
                        id: "Inflationary",
                        label: "Inflationary",
                        desc: "Higher expense floor, lower net growth",
                      },
                      {
                        id: "Recession",
                        label: "Recession",
                        desc: "High jump probability and severity",
                      },
                    ].map((item) => (
                      <ListBox.Item
                        key={item.id}
                        id={item.id}
                        textValue={item.label}
                        className="flex flex-col px-3 py-2 rounded-lg hover:bg-white/5 cursor-pointer outline-none data-[selected=true]:bg-neon-cyan/10 data-[selected=true]:text-neon-cyan"
                      >
                        <div className="font-bold text-sm">{item.label}</div>
                        <div className="text-[10px] opacity-60">
                          {item.desc}
                        </div>
                      </ListBox.Item>
                    ))}
                  </ListBox>
                </Select.Popover>
              </Select>
            </div>

            {/* Projection Horizon */}
            <div className="space-y-3">
              <Label className="text-white text-sm font-bold uppercase tracking-wider">
                Projection Horizon
              </Label>
              <Select
                value={String(draft.days)}
                onChange={(v) => setDraft({ ...draft, days: Number(v) })}
                className="w-full"
              >
                <Select.Trigger className="bg-white/5 border-white/10 hover:border-white/20 h-12 rounded-xl px-4 flex justify-between items-center w-full transition-colors">
                  <Select.Value className="text-white text-sm font-medium" />
                  <Clock size={16} className="text-[#5E6272]" />
                </Select.Trigger>
                <Select.Popover className="bg-[#121212] border border-white/10 rounded-xl shadow-2xl z-100">
                  <ListBox className="p-1">
                    {[30, 60, 90, 180].map((d) => (
                      <ListBox.Item
                        key={d}
                        id={String(d)}
                        textValue={`${d} Days`}
                        className="px-3 py-2 rounded-lg hover:bg-white/5 cursor-pointer outline-none data-[selected=true]:bg-neon-cyan/10 data-[selected=true]:text-neon-cyan font-bold text-sm"
                      >
                        {d} Days
                      </ListBox.Item>
                    ))}
                  </ListBox>
                </Select.Popover>
              </Select>
            </div>

            {/* Stress Test */}
            <div className="p-4 rounded-2xl bg-deep-pink/5 border border-deep-pink/20 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-deep-pink/10 flex items-center justify-center text-deep-pink shadow-[inset_0_0_15px_rgba(255,51,102,0.1)]">
                  <ShieldAlert size={20} />
                </div>
                <div>
                  <h4 className="text-white font-bold text-sm tracking-tight">
                    Major Stress Test
                  </h4>
                  <p className="text-[#8B8E98] text-[10px] font-medium">
                    Simulate extreme negative Merton jump
                  </p>
                </div>
              </div>
              <Switch
                isSelected={draft.stress_test_active}
                onChange={(v) => setDraft({ ...draft, stress_test_active: v })}
              >
                <Switch.Control className="bg-white/10 data-[selected=true]:bg-deep-pink">
                  <Switch.Thumb className="bg-white" />
                </Switch.Control>
              </Switch>
            </div>
          </Modal.Body>

          <Modal.Footer className="p-6 pt-0 flex gap-4">
            <Button
              onPress={onClose}
              className="flex-1 bg-white/5 hover:bg-white/10 text-[#8B8E98] hover:text-white font-bold h-12 rounded-xl transition-all cursor-pointer border border-white/5"
            >
              Cancel
            </Button>
            <Button
              onPress={handleApply}
              className="flex-1 bg-neon-cyan text-obsidian font-bold h-12 rounded-xl shadow-[0_0_20px_rgba(0,229,255,0.3)] hover:shadow-[0_0_30px_rgba(0,229,255,0.5)] transition-all cursor-pointer border-none flex items-center justify-center gap-2"
            >
              <Sparkles size={18} />
              Run Simulation
            </Button>
          </Modal.Footer>
        </Modal.Dialog>
      </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  );
}
