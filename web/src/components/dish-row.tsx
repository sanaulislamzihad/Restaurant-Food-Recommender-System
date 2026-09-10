"use client";

import { ChevronRight } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { DishCard, DishCardSkeleton } from "@/components/dish-card";
import { useImpressionLogger } from "@/lib/use-impressions";
import type { RecommendedItem } from "@/lib/types";

/**
 * A horizontally scrolling row of dishes.
 *
 * Native overflow scrolling with scroll-snap rather than a carousel library:
 * it keeps keyboard navigation, trackpad gestures and mobile momentum for free,
 * and there is no JavaScript to go wrong.
 */
export function DishRow({
  title,
  subtitle,
  items,
  isLoading,
  href,
  emptyMessage = "Nothing to show here yet.",
}: {
  title: string;
  subtitle?: React.ReactNode;
  items: RecommendedItem[];
  isLoading?: boolean;
  href?: string;
  emptyMessage?: string;
}) {
  const record = useImpressionLogger();

  React.useEffect(() => {
    if (items.length) record(items.map((entry) => entry.item.id));
  }, [items, record]);

  return (
    <section className="space-y-3">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight sm:text-xl">{title}</h2>
          {subtitle ? (
            <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>
          ) : null}
        </div>
        {href ? (
          <Link
            href={href}
            className="shrink-0 text-sm font-medium text-accent hover:underline"
          >
            <span className="inline-flex items-center gap-1">
              See all <ChevronRight className="size-4" aria-hidden="true" />
            </span>
          </Link>
        ) : null}
      </div>

      {isLoading ? (
        <div className="flex gap-4 overflow-hidden">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="w-56 shrink-0">
              <DishCardSkeleton />
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
          {emptyMessage}
        </p>
      ) : (
        <ul className="no-scrollbar -mx-4 flex snap-x snap-mandatory gap-4 overflow-x-auto px-4 pb-2">
          {items.map((entry) => (
            <li key={entry.item.id} className="w-56 shrink-0 snap-start">
              <DishCard item={entry.item} reason={entry.reason} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
