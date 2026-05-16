'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';

interface Detection {
  id: number;
  timestamp: number;
  probability: number;
  result: string;
}

interface StatsData {
  stats: {
    total_detections: number;
    not_allowed_count: number;
    avg_probability: number;
    min_probability: number;
    max_probability: number;
  };
  hourly: Array<{ hour: string; count: number; violations: number }>;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
const ESP32_TIMEOUT_MS = 15000;

function fmt(ts: number) {
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

// Returns confidence toward the actual result
function displayConf(probability: number, result: string): string {
  const conf = result === 'NOT_ALLOWED' ? probability : 1 - probability;
  return (conf * 100).toFixed(1) + '%';
}

export default function Home() {
  const [dark, setDark] = useState(true);
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const [latestDetection, setLatestDetection] = useState<Detection | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [serverConnected, setServerConnected] = useState(false);
  const [esp32Online, setEsp32Online] = useState(false);
  const [sleepMode, setSleepMode] = useState(true);
  const socketRef = useRef<Socket | null>(null);
  const esp32TimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestImageRef = useRef<string | null>(null);

  const resetEsp32Timer = useCallback(() => {
    setEsp32Online(true);
    if (esp32TimerRef.current) clearTimeout(esp32TimerRef.current);
    esp32TimerRef.current = setTimeout(() => setEsp32Online(false), ESP32_TIMEOUT_MS);
  }, []);

  const fetchImage = useCallback(async (id: number) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/image/${id}?t=${Date.now()}`, { cache: 'no-store' });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      if (latestImageRef.current) URL.revokeObjectURL(latestImageRef.current);
      latestImageRef.current = url;
      setLatestImage(url);
    } catch (e) { console.error('Image fetch error:', e); }
  }, []);

  const fetchDetections = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/detections?limit=20`, { cache: 'no-store' });
      const data: Detection[] = await res.json();
      setDetections(data);
      if (data.length > 0 && !latestImageRef.current) {
        setLatestDetection(data[0]);
        await fetchImage(data[0].id);
      }
    } catch (e) { console.error('Detections fetch error:', e); }
  }, [fetchImage]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/statistics?hours=24`, { cache: 'no-store' });
      const data: StatsData = await res.json();
      setStats(data);
    } catch (e) { console.error('Stats fetch error:', e); }
  }, []);

  const toggleSleep = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/sleep-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !sleepMode }),
      });
      const data = await res.json();
      setSleepMode(data.enabled);
    } catch (e) { console.error('Sleep toggle error:', e); }
  };

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/sleep-mode`)
      .then(r => r.json()).then(d => setSleepMode(d.enabled)).catch(() => {});
  }, []);

  useEffect(() => {
    socketRef.current = io(BACKEND_URL);
    socketRef.current.on('connect', () => setServerConnected(true));
    socketRef.current.on('disconnect', () => setServerConnected(false));
    socketRef.current.on('new-detection', (data: Detection) => {
      resetEsp32Timer();
      setLatestDetection(data);
      fetchImage(data.id);
      fetchDetections();
      fetchStats();
    });
    fetchDetections();
    fetchStats();
    return () => {
      socketRef.current?.disconnect();
      if (esp32TimerRef.current) clearTimeout(esp32TimerRef.current);
      if (latestImageRef.current) URL.revokeObjectURL(latestImageRef.current);
    };
  }, [fetchDetections, fetchStats, fetchImage, resetEsp32Timer]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  const total = stats?.stats?.total_detections ?? 0;
  const violations = stats?.stats?.not_allowed_count ?? 0;
  const allowed = total - violations;
  const violationRate = total > 0 ? ((violations / total) * 100).toFixed(1) : '0.0';
  const avgConf = stats?.stats?.avg_probability ? (stats.stats.avg_probability * 100).toFixed(1) : '0.0';
  const isViolation = latestDetection?.result === 'NOT_ALLOWED';
  const pieData = [
    { name: 'Allowed', value: allowed },
    { name: 'Not Allowed', value: violations },
  ];
  const tooltipStyle = {
    backgroundColor: dark ? '#1a1a1a' : '#ffffff',
    border: `1px solid ${dark ? '#2e2e2e' : '#e5e7eb'}`,
    borderRadius: '8px',
    color: dark ? '#e8e8e8' : '#111',
    fontSize: '11px',
    fontFamily: 'DM Mono, monospace',
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        [data-theme='dark'] {
          --bg:      #0c0c0c;
          --bg2:     #141414;
          --bg3:     #1c1c1c;
          --border:  #202020;
          --border2: #2c2c2c;
          --text:    #f0f0f0;
          --text2:   #888;
          --text3:   #444;
          --accent:  #f59e0b;
          --acc-bg:  rgba(245,158,11,0.08);
          --red:     #ef4444;
          --red-bg:  rgba(239,68,68,0.08);
          --green:   #22c55e;
          --grn-bg:  rgba(34,197,94,0.08);
          --blue:    #60a5fa;
        }

        [data-theme='light'] {
          --bg:      #f5f4f0;
          --bg2:     #ffffff;
          --bg3:     #f0ede6;
          --border:  #e6e2d9;
          --border2: #d5d0c5;
          --text:    #111;
          --text2:   #666;
          --text3:   #aaa;
          --accent:  #d97706;
          --acc-bg:  rgba(217,119,6,0.06);
          --red:     #dc2626;
          --red-bg:  rgba(220,38,38,0.05);
          --green:   #16a34a;
          --grn-bg:  rgba(22,163,74,0.06);
          --blue:    #2563eb;
        }

        html, body {
          font-family: 'DM Sans', sans-serif;
          background: var(--bg);
          color: var(--text);
          min-height: 100vh;
          -webkit-font-smoothing: antialiased;
          overflow-x: hidden;
          width: 100%;
        }

        .shell {
          width: 100%;
          max-width: 1280px;
          margin: 0 auto;
          padding: 0 16px 56px;
        }

        /* ── Header ── */
        .hdr {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 0;
          border-bottom: 1px solid var(--border);
          margin-bottom: 20px;
          gap: 12px;
        }
        .hdr-l { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .hdr-icon {
          width: 40px; height: 40px; flex-shrink: 0;
          border-radius: 11px;
          background: var(--acc-bg);
          border: 1px solid var(--border2);
          display: flex; align-items: center; justify-content: center;
          font-size: 20px;
        }
        .hdr-name { font-size: 16px; font-weight: 600; letter-spacing: -.3px; }
        .hdr-sub  { font-size: 11px; color: var(--text2); margin-top: 1px; }
        .hdr-r {
          display: flex; align-items: center;
          gap: 8px; flex-wrap: wrap;
          justify-content: flex-end;
        }

        /* ── Status indicators ── */
        .status-item {
          display: flex; align-items: center; gap: 6px;
          font-size: 12px; font-weight: 500;
          color: var(--text2); white-space: nowrap;
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .dot-on   { background: var(--green); box-shadow: 0 0 5px var(--green); }
        .dot-off  { background: var(--text3); }
        .dot-warn { background: var(--red);   box-shadow: 0 0 5px var(--red); }

        /* ── Icon button ── */
        .icon-btn {
          width: 40px; height: 40px;
          border-radius: 10px;
          border: 1px solid var(--border2);
          background: var(--bg2);
          color: var(--text2);
          cursor: pointer;
          display: flex; align-items: center; justify-content: center;
          font-size: 18px;
          -webkit-tap-highlight-color: transparent;
          transition: background .15s;
          flex-shrink: 0;
        }
        .icon-btn:active { background: var(--bg3); }

        /* ── Sleep toggle ── */
        .sleep-toggle {
          display: flex; align-items: center; gap: 10px;
          padding: 8px 14px;
          border-radius: 10px;
          border: 1px solid var(--border2);
          background: var(--bg2);
          cursor: pointer;
          -webkit-tap-highlight-color: transparent;
          transition: background .15s;
          flex-shrink: 0;
        }
        .sleep-toggle:active { background: var(--bg3); }
        .sleep-label { font-size: 13px; font-weight: 500; color: var(--text2); white-space: nowrap; }
        .switch { position: relative; width: 42px; height: 24px; flex-shrink: 0; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
          position: absolute; inset: 0;
          border-radius: 24px; background: var(--border2);
          transition: background .2s; cursor: pointer;
        }
        .slider::before {
          content: '';
          position: absolute;
          width: 18px; height: 18px;
          border-radius: 50%; background: white;
          left: 3px; top: 3px;
          transition: transform .2s;
          box-shadow: 0 1px 3px rgba(0,0,0,.3);
        }
        input:checked + .slider { background: var(--green); }
        input:checked + .slider::before { transform: translateX(18px); }

        /* ── Alert ── */
        .alert {
          display: flex; align-items: flex-start; gap: 12px;
          padding: 14px 16px;
          border-radius: 10px;
          border: 1px solid var(--red);
          background: var(--red-bg);
          margin-bottom: 20px;
          animation: fadein .3s ease;
        }
        @keyframes fadein {
          from { opacity: 0; transform: translateY(-4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .alert-title { font-size: 13px; font-weight: 600; color: var(--red); }
        .alert-sub   { font-size: 12px; color: var(--text2); margin-top: 2px; }

        /* ── Card ── */
        .card {
          background: var(--bg2);
          border: 1px solid var(--border);
          border-radius: 12px;
          overflow: hidden;
        }
        .card-hdr {
          padding: 12px 16px;
          border-bottom: 1px solid var(--border);
          background: var(--bg3);
          display: flex; align-items: center; justify-content: space-between;
        }
        .card-title {
          font-size: 11px; font-weight: 600;
          letter-spacing: .07em; text-transform: uppercase;
          color: var(--text2);
        }
        .card-meta { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--text3); }

        /* ── Feed ── */
        .feed {
          aspect-ratio: 4/3; background: #000;
          display: flex; align-items: center; justify-content: center;
          position: relative; overflow: hidden; width: 100%;
        }
        .feed img { width: 100%; height: 100%; object-fit: contain; display: block; }
        .feed-empty {
          display: flex; flex-direction: column; align-items: center; gap: 10px;
          color: var(--text3);
        }
        .feed-empty-ico { font-size: 28px; opacity: .3; }
        .feed-empty-txt { font-size: 12px; }
        .feed-overlay {
          position: absolute; bottom: 10px; right: 10px;
          font-family: 'DM Mono', monospace; font-size: 11px;
          background: rgba(0,0,0,.55); color: #eee;
          padding: 3px 8px; border-radius: 5px;
          backdrop-filter: blur(4px);
        }

        /* ── Stats grid — 2 cols on mobile, column on desktop ── */
        .stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-top: 10px;
        }
        .stat {
          background: var(--bg2);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 14px 16px;
        }
        .stat-lbl {
          font-size: 10px; font-weight: 600;
          text-transform: uppercase; letter-spacing: .07em;
          color: var(--text2); margin-bottom: 6px;
        }
        .stat-val {
          font-family: 'DM Mono', monospace;
          font-size: 26px; font-weight: 500;
          line-height: 1; letter-spacing: -1px;
        }
        .c-text { color: var(--text); }
        .c-red  { color: var(--red); }
        .c-acc  { color: var(--accent); }
        .c-blue { color: var(--blue); }

        /* ── Section spacing ── */
        .section { margin-top: 10px; }

        /* ── Chart ── */
        .chart-wrap { height: 180px; }

        /* ── Pie legend ── */
        .pie-leg { display: flex; justify-content: center; gap: 20px; margin-top: 10px; }
        .pie-leg-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }
        .pie-leg-dot { width: 8px; height: 8px; border-radius: 50%; }

        /* ── Table ── */
        .tbl-wrap { overflow-x: hidden; width: 100%; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        thead th {
          padding: 10px 14px; text-align: left;
          font-size: 10px; font-weight: 600;
          text-transform: uppercase; letter-spacing: .08em;
          color: var(--text3);
          border-bottom: 1px solid var(--border);
          background: var(--bg3);
        }
        tbody tr { border-bottom: 1px solid var(--border); }
        tbody tr:last-child { border-bottom: none; }
        tbody td { padding: 10px 14px; font-size: 12px; color: var(--text2); }
        .mono { font-family: 'DM Mono', monospace; font-size: 11px; }
        .badge {
          display: inline-flex; align-items: center; gap: 5px;
          padding: 3px 8px; border-radius: 20px;
          font-size: 11px; font-weight: 500;
        }
        .badge-red { background: var(--red-bg); color: var(--red); border: 1px solid rgba(239,68,68,.18); }
        .badge-grn { background: var(--grn-bg); color: var(--green); border: 1px solid rgba(34,197,94,.18); }
        .bdot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
        .tbl-empty td { text-align: center; padding: 40px; color: var(--text3); font-size: 12px; }

        /* ── Desktop ── */
        @media (min-width: 768px) {
          .shell { padding: 0 24px 56px; }
          .hdr { padding: 20px 0; margin-bottom: 24px; }
          .grid-main {
            display: grid;
            grid-template-columns: 1fr 260px;
            gap: 14px;
            margin-bottom: 14px;
          }
          .stats-grid {
            display: flex; flex-direction: column;
            gap: 10px; margin-top: 0;
          }
          .stat-val { font-size: 28px; }
          .grid-bot {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
          }
          .section { margin-top: 14px; }
          .chart-wrap { height: 200px; }
        }
      `}</style>

      <div className="shell">
        {/* Header */}
        <header className="hdr">
          <div className="hdr-l">
            <div className="hdr-icon">🐕</div>
            <div>
              <div className="hdr-name">DogCam</div>
              <div className="hdr-sub">Live monitoring</div>
            </div>
          </div>
          <div className="hdr-r">
            <div className="status-item">
              <div className={`dot ${serverConnected ? 'dot-on' : 'dot-off'}`} />
              Server
            </div>
            <div className="status-item">
              <div className={`dot ${esp32Online ? 'dot-on' : 'dot-warn'}`} />
              Camera
            </div>
            <div className="sleep-toggle" onClick={toggleSleep}>
              <span className="sleep-label">Sleep</span>
              <label className="switch" onClick={e => e.stopPropagation()}>
                <input type="checkbox" checked={sleepMode} onChange={toggleSleep} />
                <span className="slider" />
              </label>
            </div>
            <button className="icon-btn" onClick={() => setDark(d => !d)}>
              {dark ? '☀️' : '🌙'}
            </button>
          </div>
        </header>

        {/* Alert */}
        {isViolation && latestDetection && (
          <div className="alert">
            <span style={{ fontSize: 16 }}>⚠️</span>
            <div>
              <div className="alert-title">Intrusion detected</div>
              <div className="alert-sub">
                {displayConf(latestDetection.probability, latestDetection.result)} confidence · {fmt(latestDetection.timestamp)}
              </div>
            </div>
          </div>
        )}

        {/* Main grid */}
        <div className="grid-main">
          {/* Feed */}
          <div className="card">
            <div className="card-hdr">
              <span className="card-title">Live feed</span>
              {latestDetection && (
                <span className="card-meta">#{latestDetection.id} · {fmt(latestDetection.timestamp)}</span>
              )}
            </div>
            <div className="feed">
              {latestImage ? (
                <>
                  <img src={latestImage} alt="Latest frame" key={latestDetection?.id} />
                  {latestDetection && (
                    <div className="feed-overlay">
                      {displayConf(latestDetection.probability, latestDetection.result)} · {latestDetection.result === 'NOT_ALLOWED' ? 'Not allowed' : 'Allowed'}
                    </div>
                  )}
                </>
              ) : (
                <div className="feed-empty">
                  <div className="feed-empty-ico">📷</div>
                  <div className="feed-empty-txt">Waiting for camera…</div>
                </div>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="stats-grid">
            {[
              { label: 'Total detections', value: total,               cls: 'c-text' },
              { label: 'Violations',       value: violations,          cls: 'c-red'  },
              { label: 'Violation rate',   value: violationRate + '%', cls: 'c-acc'  },
              { label: 'Avg confidence',   value: avgConf + '%',       cls: 'c-blue' },
            ].map(s => (
              <div className="stat" key={s.label}>
                <div className="stat-lbl">{s.label}</div>
                <div className={`stat-val ${s.cls}`}>{s.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Charts */}
        <div className="grid-bot section">
          <div className="card">
            <div className="card-hdr"><span className="card-title">Hourly activity</span></div>
            <div style={{ padding: '14px 10px 10px' }}>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats?.hourly ?? []} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                    <XAxis dataKey="hour" stroke="var(--text3)" fontSize={9} tickLine={false} axisLine={false} />
                    <YAxis stroke="var(--text3)" fontSize={9} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: 'var(--border2)' }} />
                    <Line type="monotone" dataKey="count"      stroke="var(--blue)" strokeWidth={2} dot={false} name="Total" />
                    <Line type="monotone" dataKey="violations" stroke="var(--red)"  strokeWidth={2} dot={false} name="Violations" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card section">
            <div className="card-hdr"><span className="card-title">Detection split</span></div>
            <div style={{ padding: '14px 10px 10px' }}>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={72} paddingAngle={3} dataKey="value" strokeWidth={0}>
                      <Cell fill="#22c55e" />
                      <Cell fill="#ef4444" />
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="pie-leg">
                <div className="pie-leg-item">
                  <div className="pie-leg-dot" style={{ background: '#22c55e' }} />
                  Allowed ({allowed})
                </div>
                <div className="pie-leg-item">
                  <div className="pie-leg-dot" style={{ background: '#ef4444' }} />
                  Not allowed ({violations})
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="card section">
          <div className="card-hdr">
            <span className="card-title">Recent detections</span>
            <span className="card-meta">Last {detections.length} events</span>
          </div>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th style={{ width: '35%' }}>Time</th>
                  <th style={{ width: '40%' }}>Result</th>
                  <th style={{ width: '25%' }}>Conf.</th>
                </tr>
              </thead>
              <tbody>
                {detections.length === 0 ? (
                  <tr className="tbl-empty"><td colSpan={3}>No detections yet…</td></tr>
                ) : detections.map(det => (
                  <tr key={det.id}>
                    <td className="mono">{fmt(det.timestamp)}</td>
                    <td>
                      <span className={`badge ${det.result === 'NOT_ALLOWED' ? 'badge-red' : 'badge-grn'}`}>
                        <span className="bdot" />
                        {det.result === 'NOT_ALLOWED' ? 'Not allowed' : 'Allowed'}
                      </span>
                    </td>
                    <td className="mono">{displayConf(det.probability, det.result)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}
