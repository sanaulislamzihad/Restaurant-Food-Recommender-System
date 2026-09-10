"use client";

import { useQuery } from "@tanstack/react-query";
import { SlidersHorizontal, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { DishCard, DishCardSkeleton } from "@/components/dish-card";
import { Button } from "@/components/ui/button";
import {
  Alert,
  Card,
  EmptyState,
  Input,
  Label,
  Select,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { SPICE_LABELS, cuisineStyle } from "@/lib/cuisine";
import type { MenuQuery, MenuSort } from "@/lib/types";
import { useImpressionLogger } from "@/lib/use-impressions";
import { cn, humanise } from "@/lib/utils";

const PAGE_SIZE = 24;

const SORTS: Array<{ value: MenuSort; label: string }> = [
  { value: "popularity", label: "Most popular" },
  { value: "rating", label: "Highest rated" },
  { value: "price_asc", label: "Price: low to high" },
  { value: "price_desc", label: "Price: high to low" },
  { value: "newest", label: "Newest" },
  { value: "name", label: "Name (A–Z)" },
];

/**
 * Filters live in the URL rather than component state, so a filtered menu can
 * be bookmarked, shared, and survives the back button.
 */
function useMenuFilters() {
  const params = useSearchParams();
  const router = useRouter();

  const filters = React.useMemo(() => {
    const cuisine = params.getAll("cuisine");
    const sort = (params.get("sort") as MenuSort | null) ?? "popularity";
    const query: MenuQuery = {
      cuisine: cuisine.length ? cuisine : undefined,
      search: params.get("search") ?? undefined,
      max_spice: params.get("max_spice") ? Number(params.get("max_spice")) : undefined,
      min_price: params.get("min_price") ? Number(params.get("min_price")) : undefined,
      max_price: params.get("max_price") ? Number(params.get("max_price")) : undefined,
      is_veg: params.get("is_veg") === "true" ? true : undefined,
      sort,
      limit: PAGE_SIZE,
      offset: params.get("offset") ? Number(params.get("offset")) : 0,
    };
    return query;
  }, [params]);

  const update = React.useCallback(
    (changes: Record<string, string | string[] | null>) => {
      const next = new URLSearchParams(params.toString());
      for (const [key, value] of Object.entries(changes)) {
        next.delete(key);
        if (value === null) continue;
        if (Array.isArray(value)) for (const entry of value) next.append(key, entry);
        else next.set(key, value);
      }
      // Any filter change invalidates the current page position.
      if (!("offset" in changes)) next.delete("offset");
      router.push(`/menu?${next.toString()}`, { scroll: false });
    },
    [params, router],
  );

  return { filters, update };
}

function MenuContent() {
  const { filters, update } = useMenuFilters();
  const record = useImpressionLogger();
  const [searchDraft, setSearchDraft] = React.useState(filters.search ?? "");

  React.useEffect(() => setSearchDraft(filters.search ?? ""), [filters.search]);

  const cuisines = useQuery({ queryKey: ["cuisines"], queryFn: () => api.cuisines() });
  const menu = useQuery({
    queryKey: ["menu", filters],
    queryFn: () => api.menu(filters),
  });

  React.useEffect(() => {
    if (menu.data?.items.length) record(menu.data.items.map((item) => item.id));
  }, [menu.data, record]);

  const selectedCuisines = filters.cuisine ?? [];
  const activeCount =
    selectedCuisines.length +
    (filters.is_veg ? 1 : 0) +
    (filters.max_spice !== undefined ? 1 : 0) +
    (filters.min_price !== undefined ? 1 : 0) +
    (filters.max_price !== undefined ? 1 : 0);

  const offset = filters.offset ?? 0;

  // `update` already carries the other params forward, so paging only has to
  // set the offset.
  const goToOffset = React.useCallback(
    (next: number) => {
      update({ offset: String(next) });
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [update],
  );
  const total = menu.data?.total ?? 0;
  const shown = menu.data?.items.length ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Menu</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {menu.isLoading
              ? "Loading dishes…"
              : `${total} ${total === 1 ? "dish" : "dishes"}${
                  activeCount ? " matching your filters" : ""
                }`}
          </p>
        </div>

        <div className="flex items-end gap-2">
          <div className="w-40">
            <Label htmlFor="sort">Sort</Label>
            <Select
              id="sort"
              className="mt-1"
              value={filters.sort}
              onChange={(event) => update({ sort: event.target.value })}
            >
              {SORTS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
        <aside className="space-y-5 lg:sticky lg:top-20 lg:self-start">
          <Card className="space-y-5 p-4">
            <div className="flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-semibold">
                <SlidersHorizontal className="size-4" aria-hidden="true" />
                Filters
              </h2>
              {activeCount > 0 ? (
                <Button
                  variant="link"
                  size="sm"
                  className="h-auto p-0"
                  onClick={() =>
                    update({
                      cuisine: null,
                      is_veg: null,
                      max_spice: null,
                      min_price: null,
                      max_price: null,
                      search: null,
                    })
                  }
                >
                  Clear all
                </Button>
              ) : null}
            </div>

            <form
              onSubmit={(event) => {
                event.preventDefault();
                update({ search: searchDraft.trim() || null });
              }}
            >
              <Label htmlFor="search">Search</Label>
              <div className="mt-1 flex gap-2">
                <Input
                  id="search"
                  value={searchDraft}
                  placeholder="Kacchi, pizza…"
                  onChange={(event) => setSearchDraft(event.target.value)}
                />
                <Button size="sm" type="submit">
                  Go
                </Button>
              </div>
            </form>

            <fieldset>
              <legend className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Cuisine
              </legend>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {(cuisines.data ?? []).map((cuisine) => {
                  const active = selectedCuisines.includes(cuisine);
                  const style = cuisineStyle(cuisine);
                  return (
                    <button
                      key={cuisine}
                      type="button"
                      aria-pressed={active}
                      onClick={() =>
                        update({
                          cuisine: active
                            ? selectedCuisines.filter((entry) => entry !== cuisine)
                            : [...selectedCuisines, cuisine],
                        })
                      }
                      className={cn(
                        "rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                        active
                          ? "bg-accent text-accent-foreground"
                          : style.badge,
                      )}
                    >
                      {humanise(cuisine)}
                    </button>
                  );
                })}
              </div>
            </fieldset>

            <div>
              <Label htmlFor="max_spice">
                Maximum spice
                {filters.max_spice !== undefined
                  ? ` — ${SPICE_LABELS[filters.max_spice]}`
                  : ""}
              </Label>
              <input
                id="max_spice"
                type="range"
                min={0}
                max={5}
                step={1}
                value={filters.max_spice ?? 5}
                onChange={(event) =>
                  update({
                    max_spice:
                      event.target.value === "5" ? null : event.target.value,
                  })
                }
                className="mt-2 w-full accent-[hsl(var(--accent))]"
              />
              <p className="mt-1 text-[11px] text-muted-foreground">
                Slide to 5 to include everything.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label htmlFor="min_price">Min ৳</Label>
                <Input
                  id="min_price"
                  className="mt-1"
                  type="number"
                  min={0}
                  value={filters.min_price ?? ""}
                  onChange={(event) =>
                    update({ min_price: event.target.value || null })
                  }
                />
              </div>
              <div>
                <Label htmlFor="max_price">Max ৳</Label>
                <Input
                  id="max_price"
                  className="mt-1"
                  type="number"
                  min={0}
                  value={filters.max_price ?? ""}
                  onChange={(event) =>
                    update({ max_price: event.target.value || null })
                  }
                />
              </div>
            </div>

            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={Boolean(filters.is_veg)}
                onChange={(event) =>
                  update({ is_veg: event.target.checked ? "true" : null })
                }
                className="size-4 accent-[hsl(var(--accent))]"
              />
              Vegetarian only
            </label>
          </Card>
        </aside>

        <div className="space-y-6">
          {activeCount > 0 ? (
            <ul className="flex flex-wrap gap-2">
              {selectedCuisines.map((cuisine) => (
                <li key={cuisine}>
                  <button
                    type="button"
                    onClick={() =>
                      update({
                        cuisine: selectedCuisines.filter((e) => e !== cuisine),
                      })
                    }
                    className="inline-flex items-center gap-1 rounded-full bg-muted px-3 py-1 text-xs hover:bg-muted/70"
                  >
                    {humanise(cuisine)}
                    <X className="size-3" aria-hidden="true" />
                    <span className="sr-only">Remove filter</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          {menu.isError ? (
            <Alert>
              {menu.error instanceof Error
                ? menu.error.message
                : "Could not load the menu."}
            </Alert>
          ) : menu.isLoading ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
              {Array.from({ length: 8 }).map((_, index) => (
                <DishCardSkeleton key={index} />
              ))}
            </div>
          ) : shown === 0 ? (
            <EmptyState
              glyph="🔍"
              title="No dishes match those filters"
              description="Try widening the price range or clearing a cuisine."
              action={
                <Button
                  variant="secondary"
                  onClick={() =>
                    update({
                      cuisine: null,
                      is_veg: null,
                      max_spice: null,
                      min_price: null,
                      max_price: null,
                      search: null,
                    })
                  }
                >
                  Clear all filters
                </Button>
              }
            />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
                {menu.data!.items.map((item) => (
                  <DishCard key={item.id} item={item} />
                ))}
              </div>

              <nav
                className="flex items-center justify-between gap-4"
                aria-label="Pagination"
              >
                <Button
                  variant="secondary"
                  disabled={offset === 0}
                  onClick={() => goToOffset(Math.max(0, offset - PAGE_SIZE))}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  {offset + 1}–{offset + shown} of {total}
                </span>
                <Button
                  variant="secondary"
                  disabled={offset + shown >= total}
                  onClick={() => goToOffset(offset + PAGE_SIZE)}
                >
                  Next
                </Button>
              </nav>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function MenuPage() {
  // useSearchParams needs a Suspense boundary for Next's static rendering.
  return (
    <React.Suspense
      fallback={
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, index) => (
            <DishCardSkeleton key={index} />
          ))}
        </div>
      }
    >
      <MenuContent />
    </React.Suspense>
  );
}
