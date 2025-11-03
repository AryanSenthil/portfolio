import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from './contexts/ThemeContext'
import Portfolio from './pages/Portfolio'

const queryClient = new QueryClient()

function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <Portfolio />
      </QueryClientProvider>
    </ThemeProvider>
  )
}

export default App
