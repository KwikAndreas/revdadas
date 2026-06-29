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

    return Array.from(dataMap.values()).sort((a, b) =>
      a.date.localeCompare(b.date)
    );
  }, [historical, forecast]);

  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
            contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
            formatter={(value: any) => {
              const val = typeof value === 'number' ? value : 0;
              return [`Rp ${val.toFixed(1)} M`, ""];
            }}
            labelStyle={{ color: "#64748b", fontWeight: 600, marginBottom: 4 }}
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
