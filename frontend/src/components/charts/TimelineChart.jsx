import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function TimelineChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="colorMeetings" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f4" />
        <XAxis dataKey="month" tickLine={false} axisLine={false} fontSize={12} stroke="#9ca3af" />
        <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="#9ca3af" />
        <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #eee', fontSize: 12 }} />
        <Area type="monotone" dataKey="meetings" stroke="#6366f1" strokeWidth={2} fill="url(#colorMeetings)" />
      </AreaChart>
    </ResponsiveContainer>
  )
}
