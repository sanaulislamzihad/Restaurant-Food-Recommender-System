"use client";

import * as React from "react";

import { api } from "@/lib/api";
import { useAuth } from "@/providers/app-providers";

const FLUSH_DELAY_MS = 2_000;

/**
 * Log which dishes were put in front of the user.
 *
 * This is the negative half of the implicit-feedback training data. Without it
 * the model only ever sees what people ordered and can never learn what they
 * were offered and passed over — so the front end has to report impressions
 * even though nothing on screen depends on them.
 *
 * Calls are batched and de-duplicated per page view: one request for a whole
 * carousel rather than one per card, and a dish scrolled past twice is logged
 * once. Failures are swallowed, because losing telemetry must never break the
 * page the customer is actually looking at.
 */
export function useImpressionLogger() {
  const { user } = useAuth();
  const pending = React.useRef<Set<number>>(new Set());
  const seen = React.useRef<Set<number>>(new Set());
  const timer = React.useRef<number | null>(null);

  const flush = React.useCallback(() => {
    timer.current = null;
    const ids = [...pending.current];
    pending.current.clear();
    if (!ids.length) return;

    void api
      .logImpressions(ids.map((id) => ({ food_item_id: id, was_ordered: false })))
      .catch(() => {
        /* Telemetry is best-effort by design. */
      });
  }, []);

  const record = React.useCallback(
    (itemIds: number[]) => {
      if (!user) return; // The endpoint requires a token.
      for (const id of itemIds) {
        if (seen.current.has(id)) continue;
        seen.current.add(id);
        pending.current.add(id);
      }
      if (pending.current.size && timer.current === null) {
        timer.current = window.setTimeout(flush, FLUSH_DELAY_MS);
      }
    },
    [flush, user],
  );

  React.useEffect(() => {
    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
        flush();
      }
    };
  }, [flush]);

  return record;
}
