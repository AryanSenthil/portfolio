import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === 'dark';

  const handleClick = () => {
    console.log('Current theme:', theme);
    toggleTheme();
    console.log('Theme after toggle should be:', theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <button
      onClick={handleClick}
      className="relative inline-flex h-8 w-16 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-slate-300 dark:bg-slate-600"
      aria-label="Toggle theme"
    >
      {/* Toggle slider */}
      <span
        className={`inline-block h-6 w-6 transform rounded-full bg-white shadow-lg transition-transform ${
          isDark ? 'translate-x-9' : 'translate-x-1'
        }`}
      />

      {/* Sun icon (light mode) */}
      <Sun
        className={`absolute left-1.5 h-4 w-4 text-yellow-500 transition-opacity ${
          isDark ? 'opacity-0' : 'opacity-100'
        }`}
      />

      {/* Moon icon (dark mode) */}
      <Moon
        className={`absolute right-1.5 h-4 w-4 text-blue-300 transition-opacity ${
          isDark ? 'opacity-100' : 'opacity-0'
        }`}
      />
    </button>
  );
}
