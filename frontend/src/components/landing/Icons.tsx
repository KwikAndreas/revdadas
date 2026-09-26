/**
 * Ikon SVG khusus untuk landing page & kartu tim (pengganti lucide-react).
 * Berbasis garis (stroke), mewarisi warna teks via currentColor, dekoratif (aria-hidden).
 */
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 20, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

/** Logo RevDadas: garis denyut (pulse) di atas batang pendapatan. */
export function LogoMark({ size = 22, ...rest }: IconProps) {
  return (
    <Svg size={size} strokeWidth={2.2} {...rest}>
      <path d="M3 12.5h3.2l2.3-5.5 3.6 10 2.4-6.2 1.6 1.7H21" />
    </Svg>
  );
}

export function ArrowRightIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </Svg>
  );
}

export function ArrowDownIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 5v14" />
      <path d="m6 13 6 6 6-6" />
    </Svg>
  );
}

/** Basis data / sumber data DJPK. */
export function DatabaseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <ellipse cx="12" cy="5.5" rx="7.5" ry="2.8" />
      <path d="M4.5 5.5v6.5c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8V5.5" />
      <path d="M4.5 12v6.5c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8V12" />
    </Svg>
  );
}

/** Grafik tren dengan proyeksi putus-putus. */
export function ForecastIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 3v18h18" />
      <path d="m6.5 15 3.5-4 3 2.5 3-4.5" />
      <path d="m16 9 2.5-2.5" strokeDasharray="1.6 2.2" />
    </Svg>
  );
}

/** Radar deteksi anomali. */
export function RadarIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <path d="M12 12 18.4 5.6" />
      <circle cx="15.6" cy="9.2" r="1.1" fill="currentColor" stroke="none" />
    </Svg>
  );
}

/** Timbangan kebijakan. */
export function ScaleIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3v18" />
      <path d="M7 21h10" />
      <path d="M4.5 7h15" />
      <path d="m6.5 7-3 6.5a3 3 0 0 0 6 0Z" />
      <path d="m17.5 7-3 6.5a3 3 0 0 0 6 0Z" />
    </Svg>
  );
}

/** Gedung pemerintahan (Bapenda / Pemda). */
export function BuildingIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 21h18" />
      <path d="M4 9.5 12 4l8 5.5" />
      <path d="M6 10v8M10 10v8M14 10v8M18 10v8" />
    </Svg>
  );
}

/** Perisai audit (APIP / Inspektorat). */
export function ShieldIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3 5 6v5.5c0 4.3 3 7.8 7 9.5 4-1.7 7-5.2 7-9.5V6Z" />
      <path d="m9 12 2.2 2.2L15.5 10" />
    </Svg>
  );
}

/** Koin / bank sentral (BI & Satgas P2DD). */
export function BankIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M14.8 9.2c-.5-.9-1.6-1.4-2.8-1.4-1.6 0-2.8.8-2.8 2.1 0 2.9 5.8 1.5 5.8 4.3 0 1.3-1.3 2.1-3 2.1-1.3 0-2.4-.5-2.9-1.5" />
      <path d="M12 6.2v1.6M12 16.3v1.5" />
    </Svg>
  );
}

/** Tas kerja (sisi bisnis). */
export function BriefcaseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <path d="M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7" />
      <path d="M3 12.5h18" />
    </Svg>
  );
}

/** Target / sasaran pelanggan. */
export function TargetIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
    </Svg>
  );
}

/** Kunci / keamanan data. */
export function LockIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="4.5" y="10.5" width="15" height="10" rx="2" />
      <path d="M8 10.5V7.5a4 4 0 0 1 8 0v3" />
      <path d="M12 14.5v2.5" />
    </Svg>
  );
}

/** Tangga bertahap (jalur go-to-market). */
export function StepsIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 20h5v-5h5v-5h5V5h3" />
    </Svg>
  );
}

/** Label harga. */
export function TagIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3.5 12.3V4.5a1 1 0 0 1 1-1h7.8l8.2 8.2a1.4 1.4 0 0 1 0 2l-6.8 6.8a1.4 1.4 0 0 1-2 0Z" />
      <circle cx="8" cy="8" r="1.4" />
    </Svg>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m5 12.5 4.5 4.5L19 7.5" />
    </Svg>
  );
}

export function UsersIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="9" cy="8" r="3.5" />
      <path d="M2.5 20c.6-3.5 3.2-5.5 6.5-5.5s5.9 2 6.5 5.5" />
      <path d="M15.5 4.7a3.5 3.5 0 0 1 0 6.6" />
      <path d="M17.5 14.8c2.1.6 3.6 2.4 4 5.2" />
    </Svg>
  );
}

export function GraduationIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m2.5 9 9.5-5 9.5 5-9.5 5Z" />
      <path d="M6.5 11.2V16c1.5 1.5 3.5 2.3 5.5 2.3s4-.8 5.5-2.3v-4.8" />
      <path d="M21.5 9v5" />
    </Svg>
  );
}

export function MailIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="m3.5 6.5 8.5 6.5 8.5-6.5" />
    </Svg>
  );
}

export function GithubIcon({ size = 20, ...rest }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false" {...rest}>
      <path d="M12 2.2a9.8 9.8 0 0 0-3.1 19.1c.5.1.7-.2.7-.5v-1.7c-2.7.6-3.3-1.3-3.3-1.3-.4-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.3-1.1.6-1.3-2.2-.3-4.5-1.1-4.5-4.9 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.6 0 0 .8-.3 2.7 1a9.3 9.3 0 0 1 4.9 0c1.9-1.3 2.7-1 2.7-1 .5 1.3.2 2.3.1 2.6.6.7 1 1.6 1 2.7 0 3.8-2.3 4.6-4.5 4.9.4.3.7.9.7 1.8v2.7c0 .3.2.6.7.5A9.8 9.8 0 0 0 12 2.2Z" />
    </svg>
  );
}

export function ExternalIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M14 4h6v6" />
      <path d="M20 4 11 13" />
      <path d="M18 14v4.5A1.5 1.5 0 0 1 16.5 20h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10" />
    </Svg>
  );
}

export function PrinterIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M7 9V3.5h10V9" />
      <rect x="3.5" y="9" width="17" height="8" rx="2" />
      <path d="M7 14h10v6.5H7Z" />
    </Svg>
  );
}
