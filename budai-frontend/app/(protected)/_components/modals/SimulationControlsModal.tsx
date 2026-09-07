
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
  Switch,
} from "@heroui/react";
import { TrendingUp, Clock, ChevronDown, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

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

  const handleClose = () => {
    setDraft(initialValues);
    onClose();
  };

  const MACRO_STATES = ["Stable", "Inflationary", "Recession"];

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={(open) => !open && handleClose()} variant="blur">
        <Modal.Container placement="center" >
          <Modal.Dialog 
            className={cn(
              "relative max-w-md w-full pointer-events-auto bg-black/90 backdrop-blur-3xl border shadow-2xl overflow-hidden rounded-2xl transition-colors duration-500",
              draft.stress_test_active ? "border-danger/30" : "border-white/5"
            )}
          >
            <Modal.Header className="flex justify-center p-8 border-b border-white/5">
              <h3 className="text-[10px] font-black uppercase tracking-[0.4em] italic text-primary m-0 text-center">
                Forecast Overrides
              </h3>
            </Modal.Header>

            <Modal.Body className="p-8 space-y-10">
              {/* Section 1: Behavioral Metrics */}
              <div className="space-y-6 relative">
                <div className="absolute -top-3 left-0 text-[7px] uppercase tracking-widest text-foreground/20 font-bold">
                  Behavioral Metrics
                </div>
                
                <div className="space-y-4 pt-2">
                  <div className="flex justify-between items-center px-1">
                    <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                      Discipline Multiplier
                    </Label>
                    <span className="font-mono text-primary bg-primary/10 px-2 py-1 rounded-md text-xs font-bold shadow-[0_0_10px_rgba(0,242,255,0.1)]">
                      {draft.discipline_multiplier.toFixed(1)}x
                    </span>
                  </div>
                  <Slider
                    minValue={0.5}
                    maxValue={2.0}
                    step={0.1}
                    value={draft.discipline_multiplier}
                    onChange={(v) =>
                      setDraft({ ...draft, discipline_multiplier: v as number })
                    }
                    className="w-full"
                  >
                    <Slider.Track className="bg-white/5 h-1.5 rounded-full overflow-hidden">
                      <Slider.Fill className="bg-primary shadow-[0_0_15px_rgba(0,242,255,0.6)]" />
                      <Slider.Thumb className="w-5 h-5 bg-black border-2 border-primary shadow-xl cursor-pointer" />
                    </Slider.Track>
                  </Slider>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center px-1">
                    <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                      Growth Adjustment
                    </Label>
                    <span className="font-mono text-primary bg-primary/10 px-2 py-1 rounded-md text-xs font-bold shadow-[0_0_10px_rgba(0,242,255,0.1)]">
                      {(draft.drift_adjustment * 100).toFixed(1)}% / DAY
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
                    <Slider.Track className="bg-white/5 h-1.5 rounded-full overflow-hidden">
                      <Slider.Fill className="bg-primary shadow-[0_0_15px_rgba(0,242,255,0.6)]" />
                      <Slider.Thumb className="w-5 h-5 bg-black border-2 border-primary shadow-xl cursor-pointer" />
                    </Slider.Track>
                  </Slider>
                </div>
              </div>

              <div className="w-full h-px bg-white/5" />

              {/* Section 2: Macro Parameters */}
              <div className="space-y-8 relative">
                <div className="absolute -top-3 left-0 text-[7px] uppercase tracking-widest text-foreground/20 font-bold">
                  Macro Parameters
                </div>

                <div className="space-y-4 pt-2">
                  <div className="flex justify-between items-center px-1">
                    <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                      Economic State
                    </Label>
                  </div>
                  <div className="flex gap-2 w-full p-1 bg-white/5 rounded-xl border border-white/5">
                    {MACRO_STATES.map((state) => (
                      <button
                        key={state}
                        onClick={() => setDraft({ ...draft, macro_environment: state })}
                        className={cn(
                          "flex-1 py-2.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all",
                          draft.macro_environment === state
                            ? "bg-white/10 text-white shadow-sm"
                            : "text-foreground/40 hover:text-foreground/80 hover:bg-white/5"
                        )}
                      >
                        {state}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between px-1 bg-danger/5 border border-danger/10 p-4 rounded-xl">
                  <div className="flex items-center gap-3">
                    <AlertTriangle size={16} className={draft.stress_test_active ? "text-danger" : "text-foreground/20"} />
                    <div className="flex flex-col">
                      <Label className="text-[10px] font-black uppercase tracking-widest text-foreground cursor-pointer">
                        Stress Test Mode
                      </Label>
                      <span className="text-[8px] uppercase tracking-widest text-foreground/40 mt-1">
                        Applies severe variance shocks
                      </span>
                    </div>
                  </div>
                  <Switch 
                    isSelected={draft.stress_test_active} 
                    onChange={(isSelected) => setDraft({ ...draft, stress_test_active: isSelected })}
                    className={draft.stress_test_active ? "data-[selected=true]:bg-danger" : ""}
                    size="sm"
                  />
                </div>
              </div>

            </Modal.Body>

            <Modal.Footer className="p-8 pt-4 flex flex-col gap-4 border-t border-white/5">
              <Button
                onPress={handleApply}
                className={cn(
                  "w-full font-mono text-xs font-bold tracking-widest h-12 rounded-lg transition-all cursor-pointer shadow-lg",
                  draft.stress_test_active 
                    ? "bg-danger/10 text-danger border border-danger/30 hover:bg-danger/20" 
                    : "bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20"
                )}
              >
                EXECUTE SIMULATION
              </Button>
            </Modal.Footer>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
  );
}
