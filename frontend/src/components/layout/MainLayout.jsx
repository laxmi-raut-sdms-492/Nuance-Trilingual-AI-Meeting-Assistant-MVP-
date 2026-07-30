import { Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Sidebar from './Sidebar.jsx'
import Navbar from './Navbar.jsx'
import MobileNav from './MobileNav.jsx'

/**
 * Shell, ported from the design export.
 *
 * The sidebar is `fixed`, so main is offset by its width at md: rather than
 * being a flex sibling. pb-20 on mobile clears the fixed bottom tab bar.
 */
export default function MainLayout() {
  return (
    <div className="bg-background text-text-primary antialiased min-h-screen font-transcript-body">
      <Sidebar />

      <main className="flex-1 ml-0 md:ml-sidebar-width flex flex-col min-h-screen relative pb-20 md:pb-0">
        <Navbar />
        <div className="p-gutter flex flex-col gap-8 max-w-[1200px] w-full mx-auto">
          <Outlet />
        </div>
      </main>

      <MobileNav />

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgb(var(--color-surface-raised))',
            color: 'rgb(var(--color-text-primary))',
            border: '1px solid rgb(var(--color-border))',
          },
        }}
      />
    </div>
  )
}
