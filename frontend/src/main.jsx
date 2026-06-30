import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001";

const tabs = [
  { id: "queue", label: "AR Priority Queue" },
  { id: "report", label: "Underpayment Report" },
  { id: "checker", label: "Claim Checker" },
];

const money = (value, digits = 0) =>
  Number(value || 0).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
  });

const number = (value, digits = 0) =>
  Number(value || 0).toLocaleString("en-US", { maximumFractionDigits: digits });

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function App() {
  const [activeTab, setActiveTab] = useState("queue");
  const [health, setHealth] = useState(null);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    apiGet("/health").then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  return (
    <main className={dark ? "app dark" : "app"}>
      <nav className="topbar">
        <div>
          <div className="brand">WellMind AR Recovery Engine</div>
          <div className="subtle">Prioritization, variance review, and recovery signals</div>
        </div>
        <div className="topbarActions">
          <span className={health?.status === "ok" ? "status online" : "status offline"}>
            {health?.status === "ok" ? "API connected" : "API offline"}
          </span>
          <button className="iconButton" onClick={() => setDark((value) => !value)} title="Toggle theme">
            {dark ? "L" : "D"}
          </button>
        </div>
      </nav>

      <section className="hero">
        <div>
          <h1>AR Recovery Dashboard</h1>
          <p>Top-priority underpayment review queue with recovery summaries and a live claim checker.</p>
        </div>
        <div className="apiBase">{API_BASE}</div>
      </section>

      <div className="tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "queue" && <QueueTab />}
      {activeTab === "report" && <ReportTab />}
      {activeTab === "checker" && <ClaimChecker />}
    </main>
  );
}

function QueueTab() {
  const [options, setOptions] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    provider_type: "",
    state: "",
    tier: "",
    payer_type: "",
    hcpcs: "",
    min_recovery: 0,
  });

  useEffect(() => {
    apiGet("/dashboard/filter-options").then(setOptions).catch(console.error);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== "" && value !== 0) params.set(key, value);
    });
    setLoading(true);
    apiGet(`/dashboard/queue?${params.toString()}`)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [filters]);

  const rows = data?.rows || [];

  return (
    <section className="panelStack">
      <div className="filterGrid">
        <Select label="Provider Type" value={filters.provider_type} options={options?.provider_types} onChange={(provider_type) => setFilters({ ...filters, provider_type })} />
        <Select label="State" value={filters.state} options={options?.states} onChange={(state) => setFilters({ ...filters, state })} />
        <Select label="Priority Tier" value={filters.tier} options={options?.tiers} onChange={(tier) => setFilters({ ...filters, tier })} />
        <Select label="Payer Type" value={filters.payer_type} options={options?.payer_types} onChange={(payer_type) => setFilters({ ...filters, payer_type })} />
        <label className="field">
          <span>HCPCS contains</span>
          <input value={filters.hcpcs} onChange={(event) => setFilters({ ...filters, hcpcs: event.target.value })} placeholder="66984" />
        </label>
        <label className="field">
          <span>Min recovery</span>
          <input type="number" min="0" value={filters.min_recovery} onChange={(event) => setFilters({ ...filters, min_recovery: Number(event.target.value) })} />
        </label>
      </div>

      <Kpis
        items={[
          ["Claims in view", number(data?.total_rows)],
          ["Total est. recovery", money(data?.total_estimated_recovery)],
          ["Critical + High", number(data?.critical_high_count)],
          ["Avg confidence", data?.avg_confidence == null ? "-" : `${Math.round(data.avg_confidence * 100)}%`],
        ]}
      />

      <div className="split">
        <DataTable rows={rows} loading={loading} />
        <BarList
          title="Top 10 Estimated Recovery"
          rows={rows.slice(0, 10)}
          label={(row) => `${row.HCPCS_Cd} - ${row.Rndrng_Prvdr_State_Abrvtn}`}
          value={(row) => Number(row.estimated_recovery || 0)}
        />
      </div>
    </section>
  );
}

function ReportTab() {
  const [summary, setSummary] = useState(null);
  const [reports, setReports] = useState({});

  useEffect(() => {
    apiGet("/dashboard/summary").then(setSummary).catch(console.error);
    Promise.all(["state", "hcpcs", "provider", "payer"].map((name) => apiGet(`/dashboard/report/${name}?limit=15`).then((value) => [name, value.rows]))).then((entries) =>
      setReports(Object.fromEntries(entries)),
    );
  }, []);

  return (
    <section className="panelStack">
      <Kpis
        items={[
          ["Underpaid rows", number(summary?.total_underpaid_rows)],
          ["Total est. recovery", money(summary?.total_estimated_recovery)],
          ["Critical tier", number(summary?.critical_tier_rows)],
          ["Top state", summary?.top_state_by_recovery || "-"],
        ]}
      />
      <div className="chartGrid">
        <BarList title="Underpayment by State" rows={reports.state || []} label={(row) => row.provider_state} value={(row) => Number(row.total_estimated_recovery || 0)} />
        <BarList title="Underpayment by HCPCS" rows={reports.hcpcs || []} label={(row) => row.HCPCS_Cd} value={(row) => Number(row.total_estimated_recovery || 0)} />
        <BarList title="Underpayment by Provider" rows={reports.provider || []} label={(row) => row.provider_type} value={(row) => Number(row.total_estimated_recovery || 0)} />
        <BarList title="Underpayment by Payer Proxy" rows={reports.payer || []} label={(row) => row.payer_type_proxy} value={(row) => Number(row.total_estimated_recovery || 0)} />
      </div>
      <p className="disclaimer">CMS Medicare PUF 2023 portfolio demo. Estimated recovery amounts are prioritization signals, not confirmed recoverable dollars.</p>
    </section>
  );
}

function ClaimChecker() {
  const [form, setForm] = useState({
    procedure_category: "Evaluation_and_Management",
    payer_type_proxy: "Medicare_Participating",
    place_of_service: "O",
    tot_srvcs: 50,
    tot_benes: 30,
    avg_sbmtd_chrg: 250,
    review_flag: 0,
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await apiPost("/predict/recovery", form));
    } catch (err) {
      setError("Prediction unavailable. Restore model_outputs/*.pkl or rerun model training, then start FastAPI again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="checkerGrid">
      <form className="formPanel" onSubmit={submit}>
        <Select label="Procedure Category" value={form.procedure_category} options={["Evaluation_and_Management", "Musculoskeletal_Surgery", "Radiology", "Medicine_Services", "Pathology_Laboratory", "Respiratory_Cardiovascular_Surgery", "Digestive_Surgery", "Urinary_Genital_Surgery", "Nervous_System_Surgery", "Integumentary_Surgery", "Drugs", "HCPCS_G_Codes", "Other"]} onChange={(procedure_category) => setForm({ ...form, procedure_category })} />
        <Select label="Place of Service" value={form.place_of_service} options={["O", "F"]} onChange={(place_of_service) => setForm({ ...form, place_of_service })} />
        <Select label="Medicare Participation" value={form.payer_type_proxy} options={["Medicare_Participating", "Medicare_NonParticipating"]} onChange={(payer_type_proxy) => setForm({ ...form, payer_type_proxy })} />
        <Select label="Review Flag" value={String(form.review_flag)} options={["0", "1"]} onChange={(review_flag) => setForm({ ...form, review_flag: Number(review_flag) })} />
        <NumberField label="Total Services" value={form.tot_srvcs} onChange={(tot_srvcs) => setForm({ ...form, tot_srvcs })} />
        <NumberField label="Total Beneficiaries" value={form.tot_benes} onChange={(tot_benes) => setForm({ ...form, tot_benes })} />
        <NumberField label="Avg Submitted Charge" value={form.avg_sbmtd_chrg} onChange={(avg_sbmtd_chrg) => setForm({ ...form, avg_sbmtd_chrg })} />
        <button className="primaryButton" disabled={loading}>{loading ? "Analyzing..." : "Check Recovery Priority"}</button>
      </form>

      <div className="resultPanel">
        {error && <div className="emptyState">{error}</div>}
        {!error && !result && <div className="emptyState">Enter claim details and run the checker.</div>}
        {result && (
          <>
            <div className={`tierBadge ${result.priority_tier.toLowerCase()}`}>{result.priority_tier} Priority</div>
            <div className="score">{Math.round(result.recovery_probability * 100)}%</div>
            <p className="subtle">{result.estimated_recovery_signal}</p>
            <h3>Recommended Action</h3>
            <p>{result.recommended_action}</p>
            <h3>Recovery Signals</h3>
            <ul className="signalList">{result.top_risk_factors.map((factor) => <li key={factor}>{factor}</li>)}</ul>
          </>
        )}
      </div>
    </section>
  );
}

function Select({ label, value, options = [], onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function NumberField({ label, value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min="0" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function Kpis({ items }) {
  return (
    <div className="kpiGrid">
      {items.map(([label, value]) => (
        <article className="kpi" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </article>
      ))}
    </div>
  );
}

function DataTable({ rows, loading }) {
  const columns = [
    ["report_rank", "Rank"],
    ["Rndrng_Prvdr_Type", "Provider"],
    ["Rndrng_Prvdr_State_Abrvtn", "State"],
    ["HCPCS_Cd", "HCPCS"],
    ["payer_type_proxy", "Payer"],
    ["estimated_recovery", "Recovery"],
    ["Payment_Gap_Pct", "Gap %"],
    ["confidence_score", "Confidence"],
    ["priority_tier", "Tier"],
  ];

  return (
    <div className="tablePanel">
      <div className="panelTitle">Priority Queue</div>
      {loading ? <div className="emptyState">Loading queue...</div> : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr>{columns.map(([, label]) => <th key={label}>{label}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.report_rank}-${row.Rndrng_NPI}-${row.HCPCS_Cd}`}>
                  {columns.map(([key]) => (
                    <td key={key}>
                      {key === "estimated_recovery" ? money(row[key]) : key === "confidence_score" ? `${Math.round(Number(row[key] || 0) * 100)}%` : key === "Payment_Gap_Pct" ? `${number(row[key], 2)}%` : row[key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function BarList({ title, rows, label, value }) {
  const max = useMemo(() => Math.max(...rows.map(value), 1), [rows, value]);
  return (
    <div className="barPanel">
      <div className="panelTitle">{title}</div>
      {rows.length === 0 ? <div className="emptyState">No data</div> : rows.map((row, index) => {
        const current = value(row);
        return (
          <div className="barRow" key={`${label(row)}-${index}`}>
            <span>{label(row)}</span>
            <div className="barTrack"><div className="barFill" style={{ width: `${Math.max(4, (current / max) * 100)}%` }} /></div>
            <strong>{money(current)}</strong>
          </div>
        );
      })}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
