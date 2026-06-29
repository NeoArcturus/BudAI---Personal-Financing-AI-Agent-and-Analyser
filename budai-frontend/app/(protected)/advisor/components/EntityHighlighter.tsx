import React from "react";

export const EntityHighlighter = ({
  children,
}: {
  children: React.ReactNode;
}): React.ReactNode => {
  if (typeof children === "string") {
    const parts = children.split(/(£[0-9.,]+|[0-9.,]+%)/g);
    return (
      <>
        {parts.map((part, i) =>
          part.match(/^(£[0-9.,]+|[0-9.,]+%)$/) ? (
            <span
              key={i}
              className="animate-data-flash font-mono font-bold text-primary"
            >
              {part}
            </span>
          ) : (
            part
          ),
        )}
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
