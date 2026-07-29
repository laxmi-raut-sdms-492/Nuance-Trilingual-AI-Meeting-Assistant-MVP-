import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function WeeklyChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f4" />
        <XAxis dataKey="day" tickLine={false} axisLine={false} fontSize={12} stroke="#9ca3af" />
        <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="#9ca3af" />
        <Tooltip cursor={{ fill: '#f8f9fb' }} contentStyle={{ borderRadius: 12, border: '1px solid #eee', fontSize: 12 }} />
        <Bar dataKey="meetings" fill="#6366f1" radius={[6, 6, 0, 0]} barSize={26} />
      </BarChart>
    </ResponsiveContainer>
  )
}
