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
import { toBillions, formatCurrency } from "@/lib/utils";

interface ProportionChartProps {
  historical: HistoricalRecord[];
}

const COLORS = ["#1e3a5f", "#0284c7", "#10b981", "#f59e0b", "#64748b"];

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
      const name = r.Jenis_Pendapatan.toLowerCase();
      const val = toBillions(r.Realisasi);
      
      // 1. PAD
      if (name === "pendapatan asli daerah (pad)") {
        dataMap.set("Pendapatan Asli Daerah (PAD)", (dataMap.get("Pendapatan Asli Daerah (PAD)") || 0) + val);
      }
      // 2. Transfer
      else if (name === "pendapatan transfer pemerintah pusat" || name === "pendapatan transfer antar daerah" || name === "tkdd") {
        dataMap.set("Pendapatan Transfer", (dataMap.get("Pendapatan Transfer") || 0) + val);
      }
      // 3. Lainnya
      else if (name === "lain-lain pendapatan sesuai dengan ketentuan peraturan perundang-undangan" || name === "pendapatan hibah") {
        dataMap.set("Pendapatan Lainnya", (dataMap.get("Pendapatan Lainnya") || 0) + val);
      }
    });

    return Array.from(dataMap.entries())
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value); // Descending for largest at the top
  }, [historical]);

  return (
    <div style={{ width: "100%", height: isMobile ? 380 : 280, outline: "none" }}>
      <ResponsiveContainer>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: isMobile ? 10 : 35, bottom: 10 }}
          style={{ outline: "none" }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(val) => formatCurrency(Number(val) * 1e9)}
            tick={{ fontSize: 11, fill: "#64748b" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: isMobile ? 9 : 11, fill: "#334155", fontWeight: 600 }}
            width={isMobile ? 130 : 165}
            interval={0}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12, boxShadow: "0 4px 6px -1px rgba(0,0,0,0.08)" }}
            formatter={(value: any) => {
              const val = typeof value === 'number' ? value : 0;
              return [formatCurrency(val * 1e9), "Total Realisasi"];
            }}
            labelStyle={{ fontWeight: 600, color: "#1e3a5f", marginBottom: 4 }}
            cursor={{ fill: "#f8fafc" }}
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
