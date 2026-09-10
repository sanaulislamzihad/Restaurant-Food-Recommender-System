"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { FoodItemSummary } from "@/lib/types";
import { parsePrice } from "@/lib/utils";

export interface CartLine {
  item: FoodItemSummary;
  quantity: number;
}

interface CartState {
  lines: CartLine[];
  add: (item: FoodItemSummary, quantity?: number) => void;
  setQuantity: (itemId: number, quantity: number) => void;
  remove: (itemId: number) => void;
  clear: () => void;
}

export const useCart = create<CartState>()(
  persist(
    (set) => ({
      lines: [],

      add: (item, quantity = 1) =>
        set((state) => {
          const existing = state.lines.find((line) => line.item.id === item.id);
          if (existing) {
            return {
              lines: state.lines.map((line) =>
                line.item.id === item.id
                  ? { ...line, quantity: line.quantity + quantity }
                  : line,
              ),
            };
          }
          // The whole item is stored, not just its id, so the cart renders
          // without a network round trip and survives the menu changing under
          // it. Prices are re-read server-side at checkout regardless.
          return { lines: [...state.lines, { item, quantity }] };
        }),

      setQuantity: (itemId, quantity) =>
        set((state) => ({
          lines:
            quantity <= 0
              ? state.lines.filter((line) => line.item.id !== itemId)
              : state.lines.map((line) =>
                  line.item.id === itemId ? { ...line, quantity } : line,
                ),
        })),

      remove: (itemId) =>
        set((state) => ({
          lines: state.lines.filter((line) => line.item.id !== itemId),
        })),

      clear: () => set({ lines: [] }),
    }),
    { name: "foodrec.cart" },
  ),
);

export function cartCount(lines: CartLine[]): number {
  return lines.reduce((total, line) => total + line.quantity, 0);
}

/**
 * Subtotal for display only.
 *
 * The server recomputes every total from its own prices at checkout, so this
 * number never decides what anyone is charged.
 */
export function cartSubtotal(lines: CartLine[]): number {
  return lines.reduce(
    (total, line) => total + parsePrice(line.item.price) * line.quantity,
    0,
  );
}
