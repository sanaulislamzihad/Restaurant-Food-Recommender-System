"use client";

import { Clock, Flame, Leaf, Plus, Sparkles } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { DishImage } from "@/components/dish-image";
import { Badge, Card } from "@/components/ui/primitives";
import { Button } from "@/components/ui/button";
import { SPICE_LABELS, cuisineStyle } from "@/lib/cuisine";
import type { FoodItemSummary } from "@/lib/types";
import { cn, formatPrice, humanise } from "@/lib/utils";
import { useCart } from "@/store/cart";

/**
 * Why an item was recommended, shown as a chip on the card.
 *
 * The API returns this string already written; the frontend never invents an
 * explanation. If the backend has no reason to give, no chip is rendered rather
 * than a vague one being made up.
 */
export function ReasonChip({ reason }: { reason: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2 py-1 text-[11px] font-medium leading-none text-accent">
      <Sparkles className="size-3 shrink-0" aria-hidden="true" />
      <span className="truncate">{reason}</span>
    </span>
  );
}

export function DishCard({
  item,
  reason,
  className,
}: {
  item: FoodItemSummary;
  reason?: string;
  className?: string;
}) {
  const add = useCart((state) => state.add);
  const [added, setAdded] = React.useState(false);
  const style = cuisineStyle(item.cuisine);

  const onAdd = React.useCallback(
    (event: React.MouseEvent) => {
      // The whole card is a link; adding to the cart must not navigate.
      event.preventDefault();
      event.stopPropagation();
      add(item);
      setAdded(true);
      window.setTimeout(() => setAdded(false), 1200);
    },
    [add, item],
  );

  return (
    <Card
      className={cn(
        "group flex h-full flex-col overflow-hidden transition-shadow hover:shadow-md",
        className,
      )}
    >
      <Link href={`/menu/${item.id}`} className="flex h-full flex-col">
        <div className="relative">
          <DishImage
            name={item.name}
            cuisine={item.cuisine}
            imageUrl={item.image_url}
            className="aspect-[4/3] w-full"
          />
          {!item.is_available ? (
            <span className="absolute inset-0 grid place-items-center bg-background/70 text-sm font-medium">
              Currently unavailable
            </span>
          ) : null}
          <span
            className={cn(
              "absolute left-2 top-2 rounded-full px-2 py-0.5 text-[11px] font-medium",
              style.badge,
            )}
          >
            {humanise(item.cuisine)}
          </span>
        </div>

        <div className="flex flex-1 flex-col gap-2 p-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="line-clamp-2 text-sm font-semibold leading-snug">
              {item.name}
            </h3>
            <span className="shrink-0 text-sm font-semibold text-accent">
              {formatPrice(item.price)}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
            {item.is_veg ? (
              <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                <Leaf className="size-3" aria-hidden="true" /> Veg
              </Badge>
            ) : null}
            {item.spice_level > 0 ? (
              <Badge title={SPICE_LABELS[item.spice_level]}>
                <Flame className="size-3" aria-hidden="true" />
                {SPICE_LABELS[item.spice_level]}
              </Badge>
            ) : null}
            <Badge>
              <Clock className="size-3" aria-hidden="true" />
              {item.prep_time_min} min
            </Badge>
          </div>

          {reason ? (
            <div className="mt-auto pt-1">
              <ReasonChip reason={reason} />
            </div>
          ) : (
            <div className="mt-auto" />
          )}

          <Button
            size="sm"
            variant={added ? "secondary" : "primary"}
            onClick={onAdd}
            disabled={!item.is_available}
            className="mt-2 w-full"
          >
            {added ? (
              "Added"
            ) : (
              <>
                <Plus aria-hidden="true" /> Add to cart
              </>
            )}
          </Button>
        </div>
      </Link>
    </Card>
  );
}

export function DishCardSkeleton() {
  return (
    <Card className="overflow-hidden">
      <div className="shimmer aspect-[4/3] w-full bg-muted" />
      <div className="space-y-2 p-3">
        <div className="shimmer h-4 w-3/4 rounded bg-muted" />
        <div className="shimmer h-3 w-1/2 rounded bg-muted" />
        <div className="shimmer h-8 w-full rounded bg-muted" />
      </div>
    </Card>
  );
}
