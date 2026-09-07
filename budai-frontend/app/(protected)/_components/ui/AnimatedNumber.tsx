"use client";

import React, { useEffect, useRef } from "react";
import { motion, useSpring, useTransform } from "framer-motion";

interface AnimatedNumberProps {
  value: number;
  currency?: string;
  style?: "currency" | "decimal" | "percent";
  minimumFractionDigits?: number;
  maximumFractionDigits?: number;
}

export const AnimatedNumber = ({ 
  value, 
  currency = "GBP",
  style = "currency",
  minimumFractionDigits = 2,
  maximumFractionDigits = 2
}: AnimatedNumberProps) => {
  const spring = useSpring(value, { mass: 0.8, stiffness: 75, damping: 15 });
  const isFirstRender = useRef(true);
  
  const display = useTransform(spring, (current) => {
    if (style === "currency") {
      return new Intl.NumberFormat("en-GB", { 
        style: "currency", 
        currency,
        minimumFractionDigits,
        maximumFractionDigits
      }).format(current);
    } else if (style === "percent") {
      return new Intl.NumberFormat("en-GB", { 
        style: "percent",
        minimumFractionDigits,
        maximumFractionDigits
      }).format(current / 100);
    } else {
      return new Intl.NumberFormat("en-GB", { 
        style: "decimal",
        minimumFractionDigits,
        maximumFractionDigits
      }).format(current);
    }
  });

  useEffect(() => {
    if (isFirstRender.current) {
      spring.jump(value);
      isFirstRender.current = false;
    } else {
      spring.set(value);
    }
  }, [spring, value]);

  return <motion.span>{display}</motion.span>;
};
