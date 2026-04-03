/**
 * Landing page for the PuntLab web dashboard.
 *
 * Purpose: introduce the product, show the free daily slip preview position,
 * and establish the design language for later authenticated routes.
 * Scope: public homepage only.
 * Dependencies: shared layout styles from `app/globals.css`.
 */

import Link from "next/link";

const pipelineStages = [
  "Ingest fixtures, odds, team form, and injuries every morning.",
  "Research match context with deterministic analysis plus curated sports news.",
  "Rank the strongest opportunities and publish accumulator slips by subscription tier.",
] as const;

const productSignals = [
  {
    label: "Coverage",
    value: "Top UEFA leagues + NBA",
  },
  {
    label: "Publish Window",
    value: "07:00-10:00 WAT",
  },
  {
    label: "Delivery",
    value: "Telegram, API, web",
  },
] as const;

const tierCards = [
  {
    name: "Free",
    detail: "One daily accumulator with a concise rationale and today’s top angle.",
  },
  {
    name: "Plus",
    detail: "Up to ten slips per day, deeper explanations, and richer history views.",
  },
  {
    name: "Elite",
    detail: "Full analysis coverage, all generated slips, and admin-grade visibility.",
  },
] as const;

/**
 * Render the public landing page.
 *
 * @returns A public-facing homepage for the web dashboard bootstrap.
 */
export default function HomePage() {
  return (
    <div className="landing-page">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Autonomous sports research engine</p>
          <h1>Sharper pre-match slips, built from a daily analysis pipeline.</h1>
          <p className="hero-summary">
            PuntLab is an agent-first system that studies the full eligible slate,
            ranks the best opportunities, and publishes accumulator slips tuned for
            Nigerian bettors who want clarity over noise.
          </p>
          <div className="hero-actions">
            <Link className="primary-button" href="/subscribe">
              View plans
            </Link>
            <Link className="secondary-button" href="/stats">
              Track performance
            </Link>
          </div>
        </div>
        <aside className="preview-card" aria-label="Today&apos;s free slip preview">
          <div className="preview-header">
            <p>Today&apos;s free slip</p>
            <span className="preview-status">Preview slot</span>
          </div>
          <div className="preview-body">
            <strong>Publishing after the first full pipeline run.</strong>
            <p>
              This shell is ready to display the daily free accumulator once the
              API and pipeline outputs are connected.
            </p>
          </div>
          <dl className="signal-grid">
            {productSignals.map((signal) => (
              <div key={signal.label}>
                <dt>{signal.label}</dt>
                <dd>{signal.value}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </section>

      <section className="content-grid" aria-labelledby="pipeline-heading">
        <article className="content-card">
          <p className="section-label">How it works</p>
          <h2 id="pipeline-heading">The dashboard is a surface, not the product core.</h2>
          <ol className="numbered-list">
            {pipelineStages.map((stage) => (
              <li key={stage}>{stage}</li>
            ))}
          </ol>
        </article>

        <article className="content-card">
          <p className="section-label">Membership</p>
          <h2>Subscription tiers shape how much of the daily slate you unlock.</h2>
          <div className="tier-stack">
            {tierCards.map((tier) => (
              <div className="tier-card" key={tier.name}>
                <h3>{tier.name}</h3>
                <p>{tier.detail}</p>
              </div>
            ))}
          </div>
        </article>
      </section>
    </div>
  );
}
