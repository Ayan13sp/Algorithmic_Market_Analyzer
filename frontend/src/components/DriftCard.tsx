"use client";

import type { DriftReport } from "@/lib/api";

interface DriftCardProps {
  drift: DriftReport;
}

const STATUS: Record<DriftReport["status"], { label: string; dot: string; text: string }> = {
  ok: { label: "No drift", dot: "bg-emerald-400", text: "text-emerald-400" },
  warning: { label: "Warning", dot: "bg-amber-400", text: "text-amber-400" },
  drift: { label: "Drift alert", dot: "bg-red-400", text: "text-red-400" },
  insufficient_data: { label: "Not enough data", dot: "bg-white/30", text: "text-white/50" },
};

const pct = (v?: number | null) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-white/40">{label}</dt>
      <dd className="text-right">
        <span className="font-semibold text-white">{value}</span>
        {hint && <span className="ml-1.5 text-[10px] text-white/30">{hint}</span>}
      </dd>
    </div>
  );
}

/**
 * DriftCard — is the latest quarter still like the regime the model was
 * trained on? Shows the volatility regime, how many features drifted and the
 * recent out-of-sample error rate, plus any alerts.
 */
export default function DriftCard({ drift }: DriftCardProps) {
  const status = STATUS[drift.status];
  const vol = drift.volatility;
  const concept = drift.concept;

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-white/40">Drift Monitor</p>
        <span className={`flex items-center gap-1.5 text-xs font-semibold ${status.text}`}>
          <span className={`h-2 w-2 rounded-full ${status.dot}`} />
          {status.label}
        </span>
      </div>
      {drift.n_current != null && drift.n_reference != null && (
        <p className="mt-1 text-[10px] text-white/30">
          Last {drift.n_current} days vs the {drift.n_reference} before them
        </p>
      )}

      <dl className="mt-3 space-y-2 text-xs">
        {vol && (
          <Row
            label="Volatility"
            value={`${vol.current_vol_pct}% vs ${vol.reference_vol_pct}%`}
            hint={`${Math.round(vol.percentile)}th pct · ${vol.ratio != null ? `${vol.ratio.toFixed(2)}×` : vol.regime}`}
          />
        )}
        <Row
          label="Drifted features"
          value={`${drift.drifted_features.length}/${drift.features.length}`}
          hint={drift.drifted_features.slice(0, 2).join(", ") || "none"}
        />
        {concept && concept.status !== "insufficient_data" && (
          <Row
            label="Error rate"
            value={`${pct(concept.recent_error_rate)} vs ${pct(concept.reference_error_rate)}`}
            hint={concept.status}
          />
        )}
      </dl>

      {drift.alerts.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-white/5 pt-3 text-[11px] leading-relaxed text-white/50">
          {drift.alerts.map((alert) => (
            <li key={alert}>• {alert}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
