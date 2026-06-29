"use client";

import { useMemo, useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { HistoricalRecord } from "@/lib/types";
import { toBillions } from "@/lib/utils";

interface ProportionChartProps {
  historical: HistoricalRecord[];
}

const COLORS = ["#64748b", "#facc15", "#4ade80", "#dc2626", "#1e3a5f"];

export default function ProportionChart({ historical }: ProportionChartProps) {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const chartData = useMemo(() => {
    const dataMap = new Map<string, number>();

    historical.forEach((r) => {
      const current = dataMap.get(r.Jenis_Pendapatan) || 0;
      dataMap.set(r.Jenis_Pendapatan, current + toBillions(r.Realisasi));
    });

    return Array.from(dataMap.entries())
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => a.value - b.value); // Ascending for horizontal bar
  }, [historical]);

  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 20, left: isMobile ? 0 : 40, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: "#64748b" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: isMobile ? 9 : 11, fill: "#334155", fontWeight: 500 }}
            width={isMobile ? 110 : 160}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
            formatter={(value: any) => {
              const val = typeof value === 'number' ? value : 0;
              return [`Rp ${val.toFixed(1)} M`, "Realisasi"];
            }}
            labelStyle={{ display: "none" }}
            cursor={{ fill: "#f1f5f9" }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
