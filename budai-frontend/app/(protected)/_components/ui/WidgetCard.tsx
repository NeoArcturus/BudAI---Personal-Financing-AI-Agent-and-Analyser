import React from "react";
import { Card, CardProps } from "@heroui/react";
import { tv, type VariantProps } from "@heroui/styles";

export const widgetCardVariants = tv({
  base: "w-full h-full flex flex-col relative overflow-hidden",
  variants: {
    intent: {
      default: "bg-black/60 border-[1px] border-white/5",
      solid: "bg-obsidian border-[1px] border-white/5",
      transparent: "bg-transparent border-none shadow-none",
    },
    radius: {
      md: "rounded-xl",
      lg: "rounded-2xl",
      xl: "rounded-3xl",
    },
    isGlass: {
      true: "backdrop-blur-2xl backdrop-saturate-[2]",
    },
    isHoverable: {
      true: "transition-all duration-300 hover:shadow-[0_20px_40px_-20px_rgba(0,242,255,0.15)] hover:border-primary/40 hover:-translate-y-1",
    }
  },
  defaultVariants: {
    intent: "default",
    radius: "lg",
    isGlass: true,
    isHoverable: false,
  },
});

export type WidgetCardVariants = VariantProps<typeof widgetCardVariants>;

export interface WidgetCardProps extends Omit<CardProps, "radius">, WidgetCardVariants {}

export function WidgetCard({ 
  intent, 
  radius, 
  isGlass, 
  isHoverable, 
  className, 
  children,
  ...props 
}: WidgetCardProps) {
  return (
    <Card
      className={
        widgetCardVariants({ intent, radius, isGlass, isHoverable, className })
      }
      {...props}
    >
      {children}
    </Card>
  );
}
