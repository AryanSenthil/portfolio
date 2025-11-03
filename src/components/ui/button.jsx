import React from 'react';
import { motion } from 'framer-motion';

export function Button({
  children,
  className = '',
  variant = 'default',
  size = 'default',
  onClick,
  asChild = false,
  withMotion = true,
  ...props
}) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md font-medium transition-all duration-300 ease-signature focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/50 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden';

  const variants = {
    default: 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-glow-blue dark:bg-blue-500 dark:hover:bg-blue-600',
    outline: 'border border-slate-300 dark:border-slate-700 bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600',
    ghost: 'hover:bg-slate-100 dark:hover:bg-slate-800'
  };

  const sizes = {
    default: 'h-10 px-4 py-2',
    sm: 'h-9 px-3 py-2 text-sm',
    icon: 'h-10 w-10'
  };

  const combinedClassName = `${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`;

  const motionProps = withMotion ? {
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.98 },
    transition: { duration: 0.2, ease: [0.4, 0.0, 0.2, 1] }
  } : {};

  if (asChild) {
    return React.cloneElement(children, {
      className: combinedClassName,
      ...props
    });
  }

  const ButtonComponent = withMotion ? motion.button : 'button';

  return (
    <ButtonComponent
      className={combinedClassName}
      onClick={onClick}
      {...motionProps}
      {...props}
    >
      {children}
    </ButtonComponent>
  );
}
