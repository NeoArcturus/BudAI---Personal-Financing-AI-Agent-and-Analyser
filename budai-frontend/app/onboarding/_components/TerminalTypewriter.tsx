"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface TerminalTypewriterProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
  className?: string;
}

export function TerminalTypewriter({
  text,
  speed = 20,
  onComplete,
  className = "",
}: TerminalTypewriterProps) {
  const [displayedText, setDisplayedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  const onCompleteRef = React.useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    setDisplayedText("");
    setIsComplete(false);

    let currentIndex = 0;
    
    // Slight delay before typing begins
    const initialDelay = setTimeout(() => {
      const intervalId = setInterval(() => {
        if (currentIndex < text.length) {
          currentIndex++;
          setDisplayedText(text.substring(0, currentIndex));
        } else {
          clearInterval(intervalId);
          setIsComplete(true);
          if (onCompleteRef.current) onCompleteRef.current();
        }
      }, speed);

      return () => clearInterval(intervalId);
    }, 300);

    return () => clearTimeout(initialDelay);
  }, [text, speed]);

  return (
    <div className={`font-mono ${className}`}>
      {displayedText}
      <motion.span
        initial={{ opacity: 1 }}
        animate={{ opacity: isComplete ? 0 : 1 }}
        transition={{
          duration: 0.8,
          repeat: isComplete ? 0 : Infinity,
          repeatType: "reverse",
        }}
        className="inline-block ml-1 w-2.5 h-5 bg-primary align-middle shadow-[0_0_8px_rgba(0,242,255,0.8)]"
      />
    </div>
  );
}
