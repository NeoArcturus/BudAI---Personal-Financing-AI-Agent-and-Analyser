import React from "react";
import { InlineCitation } from "@/components/ai-elements/inline-citation";

export const EntityHighlighter = ({
  children,
}: {
  children: React.ReactNode;
}): React.ReactNode => {
  if (typeof children === "string") {
    const parts = children.split(/(£[0-9.,]+|[0-9.,]+%|\[tx:[^\]]+\])/g);
    return (
      <>
        {parts.map((part, i) => {
          if (part.startsWith("[tx:") && part.endsWith("]")) {
            const txId = part.slice(4, -1).trim();
            return <InlineCitation key={i} txId={txId} />;
          }
          if (part.match(/^(£[0-9.,]+|[0-9.,]+%)$/)) {
            return (
              <span
                key={i}
                className="animate-data-flash font-mono font-bold text-primary"
              >
                {part}
              </span>
            );
          }
          return part;
        })}
      </>
    );
  }
  if (Array.isArray(children)) {
    return (
      <>
        {children.map((child, i) => (
          <React.Fragment key={i}>
            <EntityHighlighter>{child}</EntityHighlighter>
          </React.Fragment>
        ))}
      </>
    );
  }
  return children;
};
