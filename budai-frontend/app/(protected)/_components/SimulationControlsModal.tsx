
"use client";

import React, { useState, useEffect } from "react";
import {
  Modal,
  Button,
  Slider,
  Select,
  ListBox,
  Label,
  Description,
} from "@heroui/react";
import { Zap, Clock, ChevronDown } from "lucide-react";

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

  const handleApply = () => {
    onApply(draft);
    onClose();
  };

  const handleClose = () => {
    setDraft(initialValues);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <Modal.Backdrop className="fixed inset-0 z-100 bg-black/40 backdrop-blur-md">
        <Modal.Container className="fixed inset-0 z-101 flex items-center justify-center p-4">
          <Modal.Dialog className="relative max-w-md w-full pointer-events-auto bg-black/60 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl overflow-hidden">
            <Modal.Header className="flex items-center gap-5 p-10 border-b-[0.5px] border-white/5 bg-white/1">
              <div className="w-12 h-12 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,242,255,0.05)]">
                <Zap size={24} />
              </div>
              <div>
                <h3 className="text-foreground font-black text-xl tracking-tighter uppercase italic m-0">
                  Simulation Parameters
                </h3>
                <p className="text-primary/50 text-[9px] font-black uppercase tracking-[0.3em] mt-1.5 m-0">
                  Adjust economy simulation
                </p>
              </div>
            </Modal.Header>

            <Modal.Body className="p-10 space-y-12">
              <div className="space-y-5">
                <div className="flex justify-between items-center px-1">
                  <Label className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                    Financing Discipline
                  </Label>
                  <span className="text-primary text-[9px] font-black uppercase tracking-[0.2em] px-3 py-1 rounded-md bg-primary/5 border-[0.5px] border-primary/20 shadow-sm">
                    {draft.discipline_multiplier < 0.8
                      ? "STRICT"
                      : draft.discipline_multiplier > 1.2
                        ? "ERRATIC"
                        : "BALANCED"}
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
                  <Slider.Track className="bg-white/5 h-1 rounded-full overflow-hidden border-[0.5px] border-white/5">
                    <Slider.Fill className="bg-primary shadow-[0_0_15px_rgba(0,242,255,0.6)]" />
                    <Slider.Thumb className="w-5 h-5 bg-foreground border-[3px] border-primary shadow-xl cursor-pointer" />
                  </Slider.Track>
                </Slider>
                <Description className="text-[9px] text-foreground/20 leading-relaxed font-bold uppercase tracking-tight pl-1">
                  Controls volatility. Lower values simulate high-predictability
                  spending cycles.
                </Description>
              </div>

              <div className="space-y-5">
                <div className="flex justify-between items-center px-1">
                  <Label className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                    Drift Delta
                  </Label>
                  <span className="text-primary font-mono text-[11px] font-black">
                    {(draft.drift_adjustment * 100).toFixed(1)}% / CYCLE
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
                  <Slider.Track className="bg-white/5 h-1 rounded-full overflow-hidden border-[0.5px] border-white/5">
                    <Slider.Fill className="bg-primary shadow-[0_0_15px_rgba(0,242,255,0.6)]" />
                    <Slider.Thumb className="w-5 h-5 bg-foreground border-[3px] border-primary shadow-xl cursor-pointer" />
                  </Slider.Track>
                </Slider>
                <Description className="text-[9px] text-foreground/20 leading-relaxed font-bold uppercase tracking-tight pl-1">
                  Offsets baseline growth rate. Target aggressive accumulation
                  via positive adjustment.
                </Description>
              </div>

              <div className="space-y-5">
                <Label className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                  Economic Environment
                </Label>
                <Select
                  value={draft.macro_environment}
                  onChange={(v) =>
                    setDraft({ ...draft, macro_environment: v as string })
                  }
                  className="w-full"
                >
                  <Select.Trigger className="bg-white/5 border-[0.5px] border-white/10 hover:border-primary/50 transition-all h-14 rounded-xl px-5 flex justify-between items-center cursor-pointer outline-none shadow-inner">
                    <Select.Value className="text-foreground text-[11px] font-black uppercase tracking-widest" />
                    <ChevronDown size={14} className="text-foreground/30" />
                  </Select.Trigger>
                  <Select.Popover
                    className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl z-100 w-65 max-w-[calc(100%-4rem)] mx-8 p-2"
                    placement="bottom"
                  >
                    <ListBox>
                      {[
                        {
                          id: "Stable",
                          label: "STABLE ECONOMY",
                          desc: "Standard historical logic parameters",
                        },
                        {
                          id: "Inflationary",
                          label: "INFLATED ECONOMY",
                          desc: "Elevated floor, lower growth rate",
                        },
                        {
                          id: "Recession",
                          label: "RECISSION ECONOMY",
                          desc: "High jump probability and severity",
                        },
                      ].map((item) => (
                        <ListBox.Item
                          key={item.id}
                          id={item.id}
                          textValue={item.label}
                          className="flex flex-col px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none data-[selected=true]:bg-primary/20 data-[selected=true]:text-primary transition-all"
                        >
                          <div className="font-black text-[11px] uppercase tracking-widest">
                            {item.label}
                          </div>
                          <div className="text-[9px] text-foreground/30 font-bold mt-1 uppercase tracking-tight">
                            {item.desc}
                          </div>
                        </ListBox.Item>
                      ))}
                    </ListBox>
                  </Select.Popover>
                </Select>
              </div>

              <div className="space-y-5">
                <Label className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                  Projection time
                </Label>
                <Select
                  value={String(draft.days)}
                  onChange={(v) => setDraft({ ...draft, days: Number(v) })}
                  className="w-full"
                >
                  <Select.Trigger className="bg-white/5 border-[0.5px] border-white/10 hover:border-primary/50 transition-all h-14 rounded-xl px-5 flex justify-between items-center cursor-pointer outline-none w-48 shadow-inner">
                    <Select.Value className="text-foreground text-[11px] font-black uppercase tracking-widest" />
                    <Clock size={16} className="text-foreground/30" />
                  </Select.Trigger>
                  <Select.Popover
                    className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl z-100 max-w-[calc(100%-4rem)] mx-8 p-2 w-48"
                    placement="bottom"
                  >
                    <ListBox>
                      {[30, 60, 90, 180].map((d) => (
                        <ListBox.Item
                          key={d}
                          id={String(d)}
                          textValue={`${d} Days`}
                          className="px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none data-[selected=true]:bg-primary/20 data-[selected=true]:text-primary font-black text-[11px] uppercase tracking-widest transition-all"
                        >
                          {d} CYCLES
                        </ListBox.Item>
                      ))}
                    </ListBox>
                  </Select.Popover>
                </Select>
              </div>
            </Modal.Body>

            <Modal.Footer className="p-10 pt-0 flex gap-4">
              <Button
                onPress={onClose}
                className="flex-1 bg-white/5 hover:bg-white/10 text-foreground/40 hover:text-foreground font-black text-[10px] uppercase tracking-widest h-14 rounded-xl transition-all cursor-pointer border-[0.5px] border-white/10"
              >
                Cancel
              </Button>
              <Button
                onPress={handleApply}
                className="flex-1 bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white border-none font-black uppercase tracking-[0.3em] text-[10px] h-14 rounded-xl shadow-[0_0_20px_rgba(0,242,255,0.3)] hover:shadow-[0_0_30px_rgba(0,242,255,0.5)] hover:scale-[1.02] transition-all cursor-pointer border-none flex items-center justify-center gap-3"
              >
                Run simulation
              </Button>
            </Modal.Footer>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  );
}
