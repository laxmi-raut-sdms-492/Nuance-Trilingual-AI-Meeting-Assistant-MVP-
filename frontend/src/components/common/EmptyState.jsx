import { Inbox } from 'lucide-react'

export default function EmptyState({ title = 'Nothing here yet', subtitle = '', icon: Icon = Inbox }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
        <Icon size={22} className="text-gray-400" />
      </div>
      <p className="text-sm font-semibold text-gray-700">{title}</p>
      {subtitle && <p className="text-xs text-gray-400 mt-1 max-w-xs">{subtitle}</p>}
    </div>
  )
}
