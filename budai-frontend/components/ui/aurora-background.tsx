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
        "relative flex flex-col h-full w-full bg-transparent transition-colors",
        className,
      )}
      {...props}
    >
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          style={
            {
              "--aurora":
                "repeating-linear-gradient(100deg,#003366 10%,#004db3 15%,#007FFF 20%,#00e5ff 25%,#8b5cf6 30%)",
              "--dark-gradient":
                "repeating-linear-gradient(100deg,#000 0%,#000 7%,transparent 10%,transparent 12%,#000 16%)",
              "--white-gradient":
                "repeating-linear-gradient(100deg,rgba(255,255,255,0.05) 0%,rgba(255,255,255,0.05) 7%,transparent 10%,transparent 12%,rgba(255,255,255,0.05) 16%)",
              "--black": "#000",
              "--transparent": "transparent",
              transform: "translateZ(0)",
              willChange: "background-position",
            } as React.CSSProperties
          }
          className={cn(
            "pointer-events-none absolute -inset-[25%] [background-image:var(--white-gradient),var(--aurora)] [background-size:300%_200%] [background-position:50%_50%] opacity-50 blur-xl filter dark:[background-image:var(--dark-gradient),var(--aurora)] animate-aurora overflow-visible",

            showRadialGradient &&
              `[mask-image:radial-gradient(ellipse_at_100%_0%,black_10%,transparent_70%)]`,
          )}
        ></div>
      </div>
      {children}
    </div>
  );
};
