"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, Megaphone, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import * as React from "react";

import { CoverageChart } from "@/components/coverage-chart";
import { Button } from "@/components/ui/button";
import { Alert, Badge, Card, Input, Label, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatPrice, formatRelative, humanise } from "@/lib/utils";
import { useAuth } from "@/providers/app-providers";

export default function AdminPage() {
  const { user, isLoading } = useAuth();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [search, setSearch] = React.useState("");

  React.useEffect(() => {
    if (!isLoading && !user) router.replace("/login?next=/admin");
  }, [isLoading, user, router]);

  const metrics = useQuery({
    queryKey: ["admin", "metrics"],
    queryFn: () => api.modelMetrics(),
    enabled: Boolean(user?.is_admin),
  });

  const coverage = useQuery({
    queryKey: ["admin", "coverage"],
    queryFn: () => api.coverage(),
    enabled: Boolean(user?.is_admin),
  });

  const menu = useQuery({
    queryKey: ["admin", "menu", search],
    queryFn: () => api.menu({ search: search || undefined, available_only: false, limit: 20 }),
    enabled: Boolean(user?.is_admin),
  });

  const retrain = useMutation({
    mutationFn: () => api.retrain({ mode: "explicit", iterations: 400, lambda: 1.0 }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "metrics"] }),
  });

  const toggle = useMutation({
    mutationFn: (input: { id: number; patch: Record<string, boolean> }) =>
      api.updateMenuItem(input.id, input.patch),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["admin"] });
      void queryClient.invalidateQueries({ queryKey: ["menu"] });
    },
  });

  if (isLoading || !user) {
    return <Skeleton className="h-64 w-full" />;
  }

  // The API enforces this; the page checks too so a customer who navigates here
  // gets an explanation rather than a wall of failed requests.
  if (!user.is_admin) {
    return (
      <div className="mx-auto max-w-md py-12 text-center">
        <h1 className="text-2xl font-bold">Staff only</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          This page needs an administrator account. Sign in as{" "}
          <code className="font-mono">admin@bhoj.example.com</code> to see it.
        </p>
      </div>
    );
  }

  const card = metrics.data?.loaded;
  const evaluation = metrics.data?.evaluation;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Admin</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Model status, catalogue coverage and menu management.
        </p>
      </header>

      {/* ---- Model card ------------------------------------------------ */}
      <section className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <Card className="space-y-4 p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold">Loaded model</h2>
              <p className="text-sm text-muted-foreground">
                Read from the artifact on disk, so this cannot disagree with what
                is actually serving.
              </p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => retrain.mutate()}
              disabled={retrain.isPending}
            >
              <RefreshCw aria-hidden="true" />
              {retrain.isPending ? "Reloading…" : "Reload / retrain"}
            </Button>
          </div>

          {metrics.isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : card ? (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
              <div>
                <dt className="text-xs uppercase text-muted-foreground">Version</dt>
                <dd className="font-medium">{card.version}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase text-muted-foreground">Trained</dt>
                <dd className="font-medium">
                  {card.trained_at ? formatRelative(card.trained_at) : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase text-muted-foreground">Neighbours</dt>
                <dd className="font-medium tabular-nums">
                  {metrics.data?.neighbours_indexed.toLocaleString()}
                </dd>
              </div>
              {Object.entries(card.training_metrics).map(([name, value]) => (
                <div key={name}>
                  <dt className="text-xs uppercase text-muted-foreground">
                    {humanise(name)}
                  </dt>
                  <dd className="font-medium tabular-nums">
                    {typeof value === "number" ? value.toFixed(4) : String(value)}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <Alert tone="info">
              No model is loaded. Run <code>python -m ml.train_cf</code> and reload.
            </Alert>
          )}

          {card ? (
            <p className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
              These are <strong>training-set</strong> figures — they say the
              optimiser converged, not that the model generalises. The held-out
              comparison is below.
            </p>
          ) : null}

          {retrain.data ? (
            <Alert tone="info">
              {retrain.data.detail}
              <pre className="mt-2 overflow-x-auto rounded bg-background p-2 font-mono text-[11px]">
                {retrain.data.command}
              </pre>
            </Alert>
          ) : null}
          {retrain.isError ? (
            <Alert>
              {retrain.error instanceof Error ? retrain.error.message : "Failed."}
            </Alert>
          ) : null}
        </Card>

        <Card className="space-y-3 p-5">
          <h2 className="text-lg font-semibold">Catalogue</h2>
          {coverage.isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : coverage.data ? (
            <dl className="space-y-2 text-sm">
              {[
                ["Dishes on the menu", coverage.data.total_items],
                ["Currently available", coverage.data.available_items],
                ["On promotion", coverage.data.promoted_items],
                ["With a similarity index", coverage.data.items_with_neighbours],
              ].map(([label, value]) => (
                <div key={String(label)} className="flex justify-between">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="font-medium tabular-nums">{String(value)}</dd>
                </div>
              ))}
            </dl>
          ) : null}
        </Card>
      </section>

      {/* ---- Coverage chart -------------------------------------------- */}
      {coverage.data ? (
        <Card className="space-y-4 p-5">
          <div>
            <h2 className="text-lg font-semibold">Recommendation coverage</h2>
            <p className="text-sm text-muted-foreground">
              How much of each cuisine collaborative filtering can actually place.
              A dish with fewer than {coverage.data.min_item_ratings} ratings is
              scored by the content model alone.
            </p>
          </div>
          <CoverageChart data={coverage.data.by_cuisine} />
        </Card>
      ) : null}

      {/* ---- Held-out evaluation --------------------------------------- */}
      {evaluation ? (
        <Card className="space-y-3 p-5">
          <div>
            <h2 className="text-lg font-semibold">Held-out evaluation</h2>
            <p className="text-sm text-muted-foreground">
              Chronological split, {formatRelative(evaluation.evaluated_at)}. Run{" "}
              <code className="font-mono text-xs">python -m ml.evaluate</code> to
              refresh.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase text-muted-foreground">
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="py-2 pr-4 text-right font-medium">RMSE</th>
                  <th className="py-2 pr-4 text-right font-medium">NDCG@10</th>
                  <th className="py-2 text-right font-medium">Coverage</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.results.map((row) => (
                  <tr key={row.name} className="border-b border-border/60 last:border-0">
                    <td className="py-2 pr-4">{row.name}</td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {row.rmse === null ? "—" : row.rmse.toFixed(4)}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {row.ndcg_at_k.toFixed(4)}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {(row.catalogue_coverage * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      {/* ---- Menu management ------------------------------------------- */}
      <Card className="space-y-4 p-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Menu</h2>
            <p className="text-sm text-muted-foreground">
              Retiring a dish hides it without deleting the row, so past orders
              still resolve.
            </p>
          </div>
          <div className="w-56">
            <Label htmlFor="admin-search">Find a dish</Label>
            <Input
              id="admin-search"
              className="mt-1"
              value={search}
              placeholder="Kacchi, pizza…"
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
        </div>

        {menu.isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase text-muted-foreground">
                  <th className="py-2 pr-4 font-medium">Dish</th>
                  <th className="py-2 pr-4 font-medium">Cuisine</th>
                  <th className="py-2 pr-4 text-right font-medium">Price</th>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(menu.data?.items ?? []).map((item) => (
                  <tr key={item.id} className="border-b border-border/60 last:border-0">
                    <td className="py-2 pr-4 font-medium">{item.name}</td>
                    <td className="py-2 pr-4 text-muted-foreground">
                      {humanise(item.cuisine)}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {formatPrice(item.price)}
                    </td>
                    <td className="py-2 pr-4">
                      {item.is_available ? (
                        <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                          Available
                        </Badge>
                      ) : (
                        <Badge className="bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300">
                          Retired
                        </Badge>
                      )}
                    </td>
                    <td className="py-2">
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={toggle.isPending}
                          onClick={() =>
                            toggle.mutate({
                              id: item.id,
                              patch: { is_promoted: true },
                            })
                          }
                        >
                          <Megaphone aria-hidden="true" /> Promote
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={toggle.isPending}
                          onClick={() =>
                            toggle.mutate({
                              id: item.id,
                              patch: { is_available: !item.is_available },
                            })
                          }
                        >
                          <Ban aria-hidden="true" />
                          {item.is_available ? "Retire" : "Restore"}
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {toggle.isError ? (
          <Alert>
            {toggle.error instanceof Error ? toggle.error.message : "Update failed."}
          </Alert>
        ) : null}
      </Card>
    </div>
  );
}
