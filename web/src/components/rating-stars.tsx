"use client";

import { Star } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/utils";

/** Read-only star display. */
export function RatingStars({
  value,
  count,
  className,
}: {
  value: number | null;
  count?: number;
  className?: string;
}) {
  if (value === null) {
    return (
      <span className={cn("text-xs text-muted-foreground", className)}>
        Not rated yet
      </span>
    );
  }
  return (
    <span className={cn("inline-flex items-center gap-1 text-xs", className)}>
      <Star className="size-3.5 fill-amber-400 text-amber-400" aria-hidden="true" />
      <span className="font-medium">{value.toFixed(1)}</span>
      {count !== undefined ? (
        <span className="text-muted-foreground">({count})</span>
      ) : null}
      <span className="sr-only">
        rated {value.toFixed(1)} out of 5
        {count !== undefined ? ` from ${count} ratings` : ""}
      </span>
    </span>
  );
}

/**
 * Interactive 1–5 star input.
 *
 * Built from real radio inputs rather than clickable divs, so it arrives with
 * keyboard support and screen-reader semantics instead of needing them bolted
 * on. The stars are the visual layer over a hidden radio group.
 */
export function RatingInput({
  value,
  onChange,
  disabled,
  name = "rating",
}: {
  value: number | null;
  onChange: (rating: number) => void;
  disabled?: boolean;
  name?: string;
}) {
  const [hovered, setHovered] = React.useState<number | null>(null);
  const shown = hovered ?? value ?? 0;

  return (
    <fieldset
      className="flex items-center gap-1"
      disabled={disabled}
      onMouseLeave={() => setHovered(null)}
    >
      <legend className="sr-only">Your rating</legend>
      {[1, 2, 3, 4, 5].map((star) => (
        <label
          key={star}
          className={cn(
            "cursor-pointer rounded p-0.5 transition-transform",
            !disabled && "hover:scale-110",
            disabled && "cursor-not-allowed opacity-60",
          )}
          onMouseEnter={() => !disabled && setHovered(star)}
        >
          <input
            type="radio"
            name={name}
            value={star}
            checked={value === star}
            onChange={() => onChange(star)}
            className="sr-only"
          />
          <Star
            className={cn(
              "size-7 transition-colors",
              star <= shown
                ? "fill-amber-400 text-amber-400"
                : "fill-transparent text-muted-foreground",
            )}
            aria-hidden="true"
          />
          <span className="sr-only">{star} stars</span>
        </label>
      ))}
    </fieldset>
  );
}
