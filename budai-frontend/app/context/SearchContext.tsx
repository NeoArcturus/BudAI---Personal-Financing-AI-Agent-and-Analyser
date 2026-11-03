"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

interface SearchContextType {
  isOpen: boolean;
  openSearch: () => void;
  closeSearch: () => void;
  toggleSearch: () => void;
}

const SearchContext = createContext<SearchContextType | undefined>(undefined);

export const SearchProvider = ({ children }: { children: React.ReactNode }) => {
  const [isOpen, setIsOpen] = useState(false);

  const openSearch = () => setIsOpen(true);
  const closeSearch = () => setIsOpen(false);
  const toggleSearch = () => setIsOpen((prev) => !prev);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac =
        typeof window !== "undefined" &&
        /Mac|iPod|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
      const isModifierActive = isMac
        ? e.metaKey && !e.ctrlKey
        : e.ctrlKey && !e.metaKey;

      if (isModifierActive && e.key.toLowerCase() === "k") {
        e.preventDefault();
        toggleSearch();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <SearchContext.Provider value={{ isOpen, openSearch, closeSearch, toggleSearch }}>
      {children}
    </SearchContext.Provider>
  );
};

export const useSearch = () => {
  const context = useContext(SearchContext);
  if (!context) throw new Error("useSearch must be used within SearchProvider");
  return context;
};
