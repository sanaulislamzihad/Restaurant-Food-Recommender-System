"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Info } from "lucide-react";
import Link from "next/link";

import { DishRow } from "@/components/dish-row";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { CUISINE_STYLES, cuisineStyle } from "@/lib/cuisine";
import { humanise } from "@/lib/utils";
import { useAuth } from "@/providers/app-providers";

export default function HomePage() {
  const { user, isLoading: authLoading } = useAuth();

  const recommended = useQuery({
    queryKey: ["recommendations", "for-me", user?.id],
    queryFn: () => api.recommendationsForMe(12),
    enabled: Boolean(user),
  });

  const popular = useQuery({
    queryKey: ["recommendations", "popular"],
    queryFn: () => api.popular(12),
  });

  const cuisines = useQuery({
    queryKey: ["cuisines"],
    queryFn: () => api.cuisines(),
  });

  const feed = recommended.data;
  // Only claim personalisation when the API says it personalised. A cold-start
  // response is a popularity list, and labelling it "picked for you" would be a
  // small lie the user can immediately detect.
  const isPersonalised = Boolean(feed && !feed.is_cold_start);

  return (
    <div className="space-y-12">
      <section className="overflow-hidden rounded-lg border border-border bg-gradient-to-br from-accent-soft to-background">
        <div className="grid gap-6 p-6 sm:p-10 md:grid-cols-[1.3fr_1fr] md:items-center">
          <div className="space-y-4">
            <p className="text-xs font-semibold uppercase tracking-widest text-accent">
              Dhaka · delivered
            </p>
            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Food chosen by what you actually order
            </h1>
            <p className="max-w-prose text-muted-foreground">
              Kacchi from Dhanmondi, pizza from Banani, mishti from wherever you
              are. The more you rate, the better the suggestions get.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link href="/menu">
                <Button size="lg">
                  Browse the menu <ArrowRight aria-hidden="true" />
                </Button>
              </Link>
              {!user && !authLoading ? (
                <Link href="/register">
                  <Button size="lg" variant="secondary">
                    Create an account
                  </Button>
                </Link>
              ) : null}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            {Object.entries(CUISINE_STYLES)
              .slice(0, 9)
              .map(([key, style]) => (
                <Link
                  key={key}
                  href={`/menu?cuisine=${key}`}
                  aria-label={`Browse ${style.label}`}
                  className={`grid aspect-square place-items-center rounded-lg bg-gradient-to-br text-3xl transition-transform hover:scale-105 ${style.gradient}`}
                >
                  <span aria-hidden="true">{style.glyph}</span>
                </Link>
              ))}
          </div>
        </div>
      </section>

      {user ? (
        <div className="space-y-3">
          <DishRow
            title={isPersonalised ? "Recommended for you" : "Popular to get you started"}
            subtitle={
              isPersonalised
                ? "Learned from your ratings and from customers with a similar taste profile."
                : "Rate a few dishes and this row becomes personal to you."
            }
            items={feed?.items ?? []}
            isLoading={recommended.isLoading}
            emptyMessage="No recommendations yet — rate a few dishes to get started."
          />

          {feed && !isPersonalised ? (
            <Card className="flex items-start gap-3 bg-muted/50 p-3 text-sm text-muted-foreground">
              <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <p>
                You have rated too few dishes for the model to have an opinion
                yet, so this row is simply what is popular. Five ratings is
                enough to switch it over.
              </p>
            </Card>
          ) : null}
        </div>
      ) : (
        <DishRow
          title="Popular right now"
          subtitle="Sign in to get a row that learns from what you order."
          items={popular.data?.items ?? []}
          isLoading={popular.isLoading}
          href="/menu"
        />
      )}

      <section className="space-y-3">
        <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
          Browse by cuisine
        </h2>
        {cuisines.isLoading ? (
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: 9 }).map((_, index) => (
              <div key={index} className="shimmer h-10 w-28 rounded-full bg-muted" />
            ))}
          </div>
        ) : (
          <ul className="flex flex-wrap gap-2">
            {(cuisines.data ?? []).map((cuisine) => {
              const style = cuisineStyle(cuisine);
              return (
                <li key={cuisine}>
                  <Link
                    href={`/menu?cuisine=${cuisine}`}
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-transform hover:scale-105 ${style.badge}`}
                  >
                    <span aria-hidden="true">{style.glyph}</span>
                    {humanise(cuisine)}
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {user ? (
        <DishRow
          title="Popular right now"
          subtitle="What everyone else is ordering this week."
          items={popular.data?.items ?? []}
          isLoading={popular.isLoading}
          href="/menu"
        />
      ) : null}
    </div>
  );
}
