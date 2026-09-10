"use client";

import type { CuisineCoverage } from "@/lib/types";
import { humanise } from "@/lib/utils";

/**
 * What share of each cuisine collaborative filtering can actually place.
 *
 * A proportion bar rather than two side-by-side bars: the question is "how much
 * of this cuisine is covered", which is one measure, and the unfilled remainder
 * carries the rest of the answer without needing a second hue competing for
 * attention.
 *
 * Hand-built for the same reasons as the taste-profile chart - nine bars do not
 * justify a charting library, and the real counts stay in the DOM as text.
 * The filled segment uses the validated `--chart-1` token; the remainder is a
 * neutral surface, not a second data colour.
 */
export function CoverageChart({ data }: { data: CuisineCoverage[] }) {
  if (data.length === 0) return null;

  return (
    <figure className="space-y-4">
      <figcaption className="sr-only">
        Share of each cuisine that collaborative filtering can rank
      </figcaption>

      {/* Two segments means the fill needs naming; colour alone would not say
          what it stands for. */}
      <ul className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <li className="flex items-center gap-1.5">
          <span className="size-3 rounded-sm bg-chart" aria-hidden="true" />
          Enough ratings to place
        </li>
        <li className="flex items-center gap-1.5">
          <span className="size-3 rounded-sm bg-muted" aria-hidden="true" />
          Content model only
        </li>
      </ul>

      <ul className="space-y-2.5">
        {data.map((row) => {
          const share = row.total_items === 0 ? 0 : row.recommendable / row.total_items;
          return (
            <li key={row.cuisine} className="space-y-1">
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="font-medium">{humanise(row.cuisine)}</span>
                <span className="tabular-nums text-muted-foreground">
                  {row.recommendable} of {row.total_items}
                  {row.cold > 0 ? <> · {row.cold} unrated</> : null}
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-l-sm bg-muted">
                <div
                  className="h-full rounded-r-[4px] bg-chart"
                  style={{ width: `${Math.max(share * 100, share > 0 ? 2 : 0)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      <table className="sr-only">
        <caption>Recommendation coverage by cuisine</caption>
        <thead>
          <tr>
            <th scope="col">Cuisine</th>
            <th scope="col">Dishes</th>
            <th scope="col">Placeable by collaborative filtering</th>
            <th scope="col">Unrated</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.cuisine}>
              <th scope="row">{humanise(row.cuisine)}</th>
              <td>{row.total_items}</td>
              <td>{row.recommendable}</td>
              <td>{row.cold}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
