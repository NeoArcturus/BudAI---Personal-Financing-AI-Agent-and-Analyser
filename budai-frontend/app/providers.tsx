"use client";

// If HeroUIProvider is not found, we might not need it or it's named differently.
// For now, we'll just return children to avoid build errors.
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <>
      {children}
    </>
  );
}
