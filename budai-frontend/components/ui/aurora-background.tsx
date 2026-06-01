"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface AuroraBackgroundProps extends React.HTMLProps<HTMLDivElement> {
  children: ReactNode;
  showRadialGradient?: boolean;
}

export const AuroraBackground = ({
  className,
  children,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) => {
  return (
    <div
      className={cn(
        "relative flex flex-col h-full w-full bg-[#030303]",
        className,
      )}
      {...props}
    >
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Layer 1: Ultra-Spectral Mesh */}
        <div
          style={{
            backgroundImage: `
              /* High-Luminosity Ultra-Spectral Mesh */
              radial-gradient(ellipse 80% 50% at 0% 0%, #00f2ff, transparent),
              radial-gradient(ellipse 80% 50% at 25% 0%, #00a3ff, transparent),
              radial-gradient(ellipse 80% 50% at 50% 0%, #0057ff, transparent),
              radial-gradient(ellipse 80% 50% at 75% 0%, #4b00ff, transparent),
              radial-gradient(ellipse 80% 50% at 100% 0%, #ad00ff, transparent),
              radial-gradient(ellipse 80% 50% at 100% 50%, #ff00ff, transparent),
              radial-gradient(ellipse 80% 50% at 100% 100%, #ff00c7, transparent),
              radial-gradient(ellipse 80% 50% at 50% 100%, #001aff, transparent),
              radial-gradient(ellipse 80% 50% at 25% 100%, #00ff85, transparent),
              radial-gradient(ellipse 80% 50% at 0% 100%, #00ffa3, transparent),
              radial-gradient(ellipse 80% 50% at 0% 50%, #007fff, transparent),
              radial-gradient(ellipse 60% 40% at 50% 50%, #3d0055, transparent)
            `,
            backgroundSize: "100% 100%",
            filter: "blur(120px) brightness(1.2) saturate(1.2)",
          }}
          className={cn(
            "pointer-events-none absolute inset-0 opacity-100 mix-blend-screen",
            showRadialGradient &&
              "[mask-image:radial-gradient(ellipse_at_center,black,transparent)]",
          )}
        ></div>

        {/* Layer 2: Prismatic Refraction Beams (Enhanced Glint) */}
        <div
          className="absolute inset-0 opacity-60 z-10"
          style={{
            backgroundImage: `
              linear-gradient(45deg, transparent 48%, rgba(255,255,255,0.15) 50%, transparent 52%),
              linear-gradient(-45deg, transparent 48%, rgba(255,255,255,0.15) 50%, transparent 52%)
            `,
          }}
        />

        {/* Layer 3: Secondary Diagonal Abyssal Rift & Corner Patches (Dual-Aurora Split) */}
        <div 
          className="absolute inset-0 z-20"
          style={{
            backgroundImage: `
              radial-gradient(circle at 100% 0%, #000000, transparent 45%),
              radial-gradient(circle at 0% 100%, #000000, transparent 45%),
              linear-gradient(225deg, transparent 10%, rgba(0,0,0,0.4) 30%, #000000 45%, #000000 55%, rgba(0,0,0,0.4) 70%, transparent 90%)
            `
          }}
        />
      </div>
      <div className="relative z-30">{children}</div>
    </div>
  );
};
