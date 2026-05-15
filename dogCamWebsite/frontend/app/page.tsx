'use client';

import { useState, useEffect, useRef } from 'react';
import io, { Socket } from 'socket.io-client';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const COLORS = ['#22c55e', '#ef4444'];

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
  hourly: Array<{
    hour: string;
    count: number;
    violations: number;
  }>;
}

export default function Home() {
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const [latestDetection, setLatestDetection] = useState<Detection | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const socketRef = useRef<Socket | null>(null);

  const BACKEND_URL = 'http://localhost:8080';

  useEffect(() => {
    socketRef.current = io(BACKEND_URL);

    socketRef.current.on('connect', () => {
      console.log('Connected to backend');
      setConnected(true);
    });

    socketRef.current.on('disconnect', () => {
      console.log('Disconnected from backend');
      setConnected(false);
    });

    socketRef.current.on('new-detection', (data: Detection) => {
      console.log('New detection:', data);
      setLatestDetection(data);
      fetchDetections();
      fetchStats();
      
      fetch(`${BACKEND_URL}/api/image/${data.id}`)
        .then(res => res.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          if (latestImage) URL.revokeObjectURL(latestImage);
          setLatestImage(url);
        })
        .catch(err => console.error('Error fetching image:', err));
    });

    fetchDetections();
    fetchStats();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (latestImage) {
        URL.revokeObjectURL(latestImage);
      }
    };
  }, []);

  const fetchDetections = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/detections?limit=20`);
      const data = await response.json();
      setDetections(data);
      
      if (data.length > 0 && data[0].id) {
        const imgResponse = await fetch(`${BACKEND_URL}/api/image/${data[0].id}`);
        const blob = await imgResponse.blob();
        const url = URL.createObjectURL(blob);
        if (latestImage) URL.revokeObjectURL(latestImage);
        setLatestImage(url);
        setLatestDetection(data[0]);
      }
    } catch (error) {
      console.error('Error fetching detections:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/statistics?hours=24`);
      const data = await response.json();
      console.log('Stats received:', data);
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const totalDetections = stats?.stats?.total_detections || 0;
  const violations = stats?.stats?.not_allowed_count || 0;
  const violationRate = totalDetections > 0 ? (violations / totalDetections * 100).toFixed(1) : 0;
  const avgConfidence = stats?.stats?.avg_probability ? (stats.stats.avg_probability * 100).toFixed(1) : 0;
  
  const pieData = [
    { name: 'Allowed', value: totalDetections - violations },
    { name: 'Not Allowed', value: violations }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🐕</span>
              <h1 className="text-xl font-semibold text-gray-900">DogCam Monitor</h1>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-500">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {latestDetection && latestDetection.result === 'NOT_ALLOWED' && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl">⚠️</span>
              <div>
                <h3 className="font-semibold text-red-800">Intrusion Detected</h3>
                <p className="text-sm text-red-600">
                  Confidence: {(latestDetection.probability * 100).toFixed(1)}% at {new Date(latestDetection.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
              <div className="border-b border-gray-200 px-4 py-3 bg-gray-50">
                <h2 className="font-medium text-gray-900">Live Camera Feed</h2>
              </div>
              <div className="relative bg-gray-900 aspect-video">
                {latestImage ? (
                  <img src={latestImage} alt="Latest detection" className="w-full h-full object-contain" />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400 mx-auto mb-3"></div>
                      <p className="text-gray-400 text-sm">Waiting for camera...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-500">Total Detections</h3>
                <span className="text-2xl font-bold text-gray-900">{totalDetections}</span>
              </div>
            </div>
            
            <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-500">Violations</h3>
                <span className="text-2xl font-bold text-red-600">{violations}</span>
              </div>
            </div>
            
            <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-500">Violation Rate</h3>
                <span className="text-2xl font-bold text-yellow-600">{violationRate}%</span>
              </div>
            </div>
            
            <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-500">Avg Confidence</h3>
                <span className="text-2xl font-bold text-blue-600">{avgConfidence}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <h3 className="font-medium text-gray-900 mb-4">Hourly Activity</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats?.hourly || []}>
                  <XAxis dataKey="hour" stroke="#9CA3AF" fontSize={12} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}
                  />
                  <Line type="monotone" dataKey="violations" stroke="#ef4444" strokeWidth={2} name="Violations" />
                  <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} name="Total" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <h3 className="font-medium text-gray-900 mb-4">Detection Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    label={(entry) => entry.name}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
          <div className="border-b border-gray-200 px-4 py-3 bg-gray-50">
            <h3 className="font-medium text-gray-900">Recent Detections</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {detections.map((det) => (
                  <tr key={det.id} className="hover:bg-gray-50 transition-colors">
                    <td className="py-2 px-4 text-sm text-gray-600">
                      {new Date(det.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2 px-4">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        det.result === 'NOT_ALLOWED' 
                          ? 'bg-red-100 text-red-800' 
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {det.result === 'NOT_ALLOWED' ? 'Not Allowed' : 'Allowed'}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-sm text-gray-600">
                      {(det.probability * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
                {detections.length === 0 && (
                  <tr>
                    <td colSpan={3} className="text-center py-8 text-gray-400 text-sm">
                      No detections yet. Waiting for camera...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
