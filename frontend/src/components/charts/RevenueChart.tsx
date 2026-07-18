import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { HistoricalRecord, ForecastRecord } from "@/lib/types";
import { toBillions } from "@/lib/utils";

interface RevenueChartProps {
  historical: HistoricalRecord[];
  forecast: ForecastRecord[];
}

export default function RevenueChart({
  historical,
  forecast,
}: RevenueChartProps) {
  const chartData = useMemo(() => {
    const dataMap = new Map<string, any>();

    // Aggregate Historical Data
    historical.forEach((r) => {
      const date = r.Tanggal.split("T")[0].substring(0, 7); // YYYY-MM
      if (!dataMap.has(date)) {
        dataMap.set(date, { date, actual: 0, forecast: null });
      }
      dataMap.get(date).actual += toBillions(r.Realisasi);
    });

    // Aggregate Forecast Data
    forecast.forEach((r) => {
      const date = r.Tanggal.split("T")[0].substring(0, 7);
      if (!dataMap.has(date)) {
        dataMap.set(date, { date, actual: null, forecast: 0 });
      }
      if (dataMap.get(date).forecast === null) {
        dataMap.get(date).forecast = 0;
      }
      dataMap.get(date).forecast += toBillions(r.Prediksi);
    });

    const sortedData = Array.from(dataMap.values()).sort((a, b) =>
      a.date.localeCompare(b.date)
    );

    // Calculate MoM percentage change
    for (let i = 1; i < sortedData.length; i++) {
      const prevActual = sortedData[i - 1].actual;
      const currActual = sortedData[i].actual;
      const prevForecast = sortedData[i - 1].forecast;
      const currForecast = sortedData[i].forecast;
      
      const prev = prevActual !== null ? prevActual : prevForecast;
      const curr = currActual !== null ? currActual : (currForecast !== null && currForecast !== 0 ? currForecast : null);
      
      if (prev !== null && curr !== null && prev > 0) {
        sortedData[i].mom = ((curr - prev) / prev) * 100;
      }
    }

    return sortedData;
  }, [historical, forecast]);

  return (
    <div style={{ width: "100%", height: 300, outline: "none" }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }} style={{ outline: "none" }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickMargin={10}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "#64748b" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(value) => `${value}`}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div style={{ background: "white", padding: 12, border: "1px solid #e2e8f0", borderRadius: 8, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)", fontSize: 12 }}>
                    <div style={{ color: "#64748b", fontWeight: 600, marginBottom: 8 }}>{label}</div>
                    {payload.map((entry: any, index: number) => (
                      <div key={index} style={{ color: entry.color, display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 4 }}>
                        <span>{entry.name}:</span>
                        <span style={{ fontWeight: 600 }}>Rp {Number(entry.value).toFixed(1)} M</span>
                      </div>
                    ))}
                    {data.mom !== null && data.mom !== undefined && (
                      <div style={{ color: data.mom >= 0 ? "#16a34a" : "#dc2626", marginTop: 8, fontSize: 11, fontWeight: 600, display: "flex", justifyContent: "flex-end" }}>
                        {data.mom >= 0 ? "▲" : "▼"} {Math.abs(data.mom).toFixed(1)}% MoM
                      </div>
                    )}
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
          <Line
            type="monotone"
            dataKey="actual"
            name="Historical Revenue (M)"
            stroke="#1e3a5f"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#1e3a5f", strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="forecast"
            name="AI Forecast (M)"
            stroke="#b91c1c"
            strokeWidth={2.5}
            strokeDasharray="5 5"
            dot={{ r: 3, fill: "#b91c1c", strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
