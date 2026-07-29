import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar.jsx'
import Navbar from './Navbar.jsx'
import { Toaster } from 'react-hot-toast'

export default function MainLayout() {
  return (
    <div className="flex min-h-screen bg-[#f8f9fb] dark:bg-gray-950">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar />
        <main className="flex-1 p-6 overflow-x-hidden">
          <Outlet />
        </main>
      </div>
      <Toaster position="top-right" />
    </div>
  )
}
