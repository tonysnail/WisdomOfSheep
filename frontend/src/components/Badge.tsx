import type { ReactNode } from 'react'

export default function Badge({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span className="badge" title={title}>
      {children}
    </span>
  )
}
