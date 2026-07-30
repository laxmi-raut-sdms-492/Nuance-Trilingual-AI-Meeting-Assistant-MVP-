export default function Loader({ label = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <div className="w-8 h-8 border-[3px] border-border border-t-primary-container rounded-full animate-spin" />
      <p className="font-meta-data text-meta-data text-text-muted">{label}</p>
    </div>
  )
}
