import Card from '../common/Card.jsx'

export default function StatCard({ label, value, icon: Icon, tint = 'primary', trend }) {
  const tints = {
    primary: 'bg-primary-50 text-primary-600',
    green: 'bg-green-50 text-green-600',
    amber: 'bg-amber-50 text-amber-600',
    pink: 'bg-pink-50 text-pink-600',
    sky: 'bg-sky-50 text-sky-600',
    teal: 'bg-teal-50 text-teal-600'
  }
  return (
    <Card className="flex items-center justify-between hover:shadow-soft transition-shadow duration-200">
      <div>
        <p className="text-xs text-gray-400 font-medium mb-1">{label}</p>
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
        {trend && <p className="text-xs text-green-500 font-medium mt-1">{trend}</p>}
      </div>
      <div className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 ${tints[tint]}`}>
        <Icon size={20} />
      </div>
    </Card>
  )
}
