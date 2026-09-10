"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Minus, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { DishImage } from "@/components/dish-image";
import { Button } from "@/components/ui/button";
import { Alert, Card, EmptyState } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { OrderResponse } from "@/lib/types";
import { formatPrice } from "@/lib/utils";
import { useAuth } from "@/providers/app-providers";
import { cartSubtotal, useCart } from "@/store/cart";

const DELIVERY_FEE = 60;

export default function CartPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { lines, setQuantity, remove, clear } = useCart();
  const [placed, setPlaced] = React.useState<OrderResponse | null>(null);

  // The store is persisted to localStorage, so the server render and the first
  // client render disagree. Waiting for mount avoids a hydration mismatch.
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  const checkout = useMutation({
    mutationFn: () =>
      api.createOrder(
        lines.map((line) => ({
          food_item_id: line.item.id,
          quantity: line.quantity,
        })),
      ),
    onSuccess: (order) => {
      setPlaced(order);
      clear();
      void queryClient.invalidateQueries({ queryKey: ["orders"] });
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      // Ordering is the strongest signal there is, so it is logged as a
      // converted impression for the implicit-feedback model.
      void api
        .logImpressions(
          order.items.map((line) => ({
            food_item_id: line.food_item_id,
            was_ordered: true,
          })),
        )
        .catch(() => {
          /* Telemetry must not fail a completed order. */
        });
    },
  });

  if (placed) {
    return (
      <div className="mx-auto max-w-xl space-y-6 py-8 text-center">
        <CheckCircle2 className="mx-auto size-14 text-success" aria-hidden="true" />
        <div>
          <h1 className="text-2xl font-bold">Order placed</h1>
          <p className="mt-2 text-muted-foreground">
            Order #{placed.id} · {formatPrice(placed.total_amount)} ·{" "}
            {placed.items.length} {placed.items.length === 1 ? "item" : "items"}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            This is a demonstration app, so no payment was taken and no food is
            on its way.
          </p>
        </div>
        <div className="flex justify-center gap-3">
          <Link href="/menu">
            <Button variant="secondary">Keep browsing</Button>
          </Link>
          <Link href="/profile">
            <Button>View your orders</Button>
          </Link>
        </div>
      </div>
    );
  }

  if (!mounted) {
    return <div className="shimmer h-64 rounded-lg bg-muted" />;
  }

  if (lines.length === 0) {
    return (
      <EmptyState
        glyph="🛒"
        title="Your cart is empty"
        description="Add a few dishes and they will show up here."
        action={
          <Link href="/menu">
            <Button>Browse the menu</Button>
          </Link>
        }
      />
    );
  }

  const subtotal = cartSubtotal(lines);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Your cart</h1>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <ul className="space-y-3">
          {lines.map((line) => (
            <li key={line.item.id}>
              <Card className="flex items-center gap-4 p-3">
                <Link href={`/menu/${line.item.id}`} className="shrink-0">
                  <DishImage
                    name={line.item.name}
                    cuisine={line.item.cuisine}
                    imageUrl={line.item.image_url}
                    className="size-20 rounded-md"
                    glyphClassName="text-3xl"
                  />
                </Link>

                <div className="min-w-0 flex-1">
                  <Link
                    href={`/menu/${line.item.id}`}
                    className="font-medium hover:text-accent"
                  >
                    {line.item.name}
                  </Link>
                  <p className="text-sm text-muted-foreground">
                    {formatPrice(line.item.price)} each
                  </p>
                </div>

                <div className="flex items-center rounded-md border border-border">
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Decrease quantity of ${line.item.name}`}
                    onClick={() => setQuantity(line.item.id, line.quantity - 1)}
                  >
                    <Minus />
                  </Button>
                  <span className="w-9 text-center text-sm font-medium">
                    {line.quantity}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Increase quantity of ${line.item.name}`}
                    onClick={() => setQuantity(line.item.id, line.quantity + 1)}
                  >
                    <Plus />
                  </Button>
                </div>

                <div className="w-20 text-right text-sm font-semibold">
                  {formatPrice(
                    Number.parseFloat(line.item.price) * line.quantity,
                  )}
                </div>

                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove ${line.item.name}`}
                  onClick={() => remove(line.item.id)}
                >
                  <Trash2 className="text-danger" />
                </Button>
              </Card>
            </li>
          ))}
        </ul>

        <Card className="h-fit space-y-4 p-5 lg:sticky lg:top-20">
          <h2 className="font-semibold">Order summary</h2>

          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-muted-foreground">Subtotal</dt>
              <dd>{formatPrice(subtotal)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted-foreground">Delivery</dt>
              <dd>{formatPrice(DELIVERY_FEE)}</dd>
            </div>
            <div className="flex justify-between border-t border-border pt-2 text-base font-semibold">
              <dt>Total</dt>
              <dd>{formatPrice(subtotal + DELIVERY_FEE)}</dd>
            </div>
          </dl>

          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Shown for reference only. The server recalculates every line from its
            own prices when the order is placed, so this figure never decides
            what anyone is charged.
          </p>

          {user ? (
            <Button
              className="w-full"
              size="lg"
              disabled={checkout.isPending}
              onClick={() => checkout.mutate()}
            >
              {checkout.isPending ? "Placing order…" : "Place order"}
            </Button>
          ) : (
            <div className="space-y-2">
              <Link href="/login">
                <Button className="w-full" size="lg">
                  Sign in to order
                </Button>
              </Link>
              <p className="text-center text-xs text-muted-foreground">
                Your cart is kept while you sign in.
              </p>
            </div>
          )}

          {checkout.isError ? (
            <Alert>
              {checkout.error instanceof Error
                ? checkout.error.message
                : "Could not place the order."}
            </Alert>
          ) : null}

          <Button variant="ghost" className="w-full" onClick={clear}>
            Clear cart
          </Button>
        </Card>
      </div>
    </div>
  );
}
