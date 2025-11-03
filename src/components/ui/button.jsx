import React from 'react';

export function Button({
  children,
  className = '',
  variant = 'default',
  size = 'default',
  onClick,
  ...props
}) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50';

  const variants = {
    default: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border border-slate-300 bg-transparent hover:bg-slate-100',
    ghost: 'hover:bg-slate-100'
  };

  const sizes = {
    default: 'h-10 px-4 py-2',
    icon: 'h-10 w-10'
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
}
