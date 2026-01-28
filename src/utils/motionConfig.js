/**
 * APPLE GLASS AESTHETIC - FRAMER MOTION CONFIGURATION
 * Sophisticated motion design utilities for academic portfolio
 * Designed for professional portfolios
 */

// ============================================
// EASING CURVES - Sophisticated & Natural
// ============================================

export const easings = {
  // General purpose smooth (Apple-style)
  signature: [0.4, 0.0, 0.2, 1],

  // Emphasis (enters, reveals)
  emphasis: [0.16, 1, 0.3, 1],

  // Quick exits
  exit: [0.4, 0.0, 1, 1],

  // Subtle hover
  hover: [0.25, 0.46, 0.45, 0.94],
};

// ============================================
// SPRING PHYSICS - Refined & Elegant
// ============================================

export const springs = {
  // Smooth spring (modals, overlays)
  smooth: {
    type: "spring",
    stiffness: 100,
    damping: 30,
    mass: 1,
  },

  // Gentle spring (cards, buttons)
  gentle: {
    type: "spring",
    stiffness: 200,
    damping: 25,
    mass: 0.8,
  },

  // Snappy spring (small elements, icons)
  snappy: {
    type: "spring",
    stiffness: 300,
    damping: 20,
    mass: 0.5,
  },

  // Bouncy (use sparingly for delight)
  bouncy: {
    type: "spring",
    stiffness: 400,
    damping: 15,
    mass: 0.6,
  },
};

// ============================================
// ANIMATION VARIANTS - Reusable Patterns
// ============================================

export const fadeInUp = {
  initial: {
    opacity: 0,
    y: 20
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: easings.emphasis,
    }
  },
  exit: {
    opacity: 0,
    y: -20,
    transition: {
      duration: 0.3,
      ease: easings.exit,
    }
  },
};

export const fadeInDown = {
  initial: {
    opacity: 0,
    y: -20
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: easings.emphasis,
    }
  },
};

export const scaleIn = {
  initial: {
    opacity: 0,
    scale: 0.95
  },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.4,
      ease: easings.emphasis,
    }
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    transition: {
      duration: 0.2,
      ease: easings.exit,
    }
  },
};

export const slideInLeft = {
  initial: {
    opacity: 0,
    x: -20
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.6,
      ease: easings.emphasis,
    }
  },
};

export const slideInRight = {
  initial: {
    opacity: 0,
    x: 20
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.6,
      ease: easings.emphasis,
    }
  },
};

// ============================================
// HOVER VARIANTS - Microinteractions
// ============================================

export const hoverLift = {
  rest: {
    y: 0,
    scale: 1,
  },
  hover: {
    y: -4,
    scale: 1.01,
    transition: {
      duration: 0.3,
      ease: easings.hover,
    }
  },
  tap: {
    scale: 0.98,
    transition: {
      duration: 0.1,
      ease: easings.exit,
    }
  },
};

export const hoverGlow = {
  rest: {
    boxShadow: "0 0 0 0px rgba(59, 130, 246, 0)",
  },
  hover: {
    boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.15), 0 4px 12px rgba(59, 130, 246, 0.2)",
    transition: {
      duration: 0.3,
      ease: easings.hover,
    }
  },
};

export const hoverScale = {
  rest: {
    scale: 1,
  },
  hover: {
    scale: 1.05,
    transition: {
      duration: 0.3,
      ease: easings.hover,
    }
  },
  tap: {
    scale: 0.95,
    transition: {
      duration: 0.1,
      ease: easings.exit,
    }
  },
};

export const iconRotate = {
  rest: {
    rotate: 0,
    scale: 1,
  },
  hover: {
    rotate: 3,
    scale: 1.1,
    transition: {
      duration: 0.3,
      ease: easings.hover,
    }
  },
};

// ============================================
// STAGGER CONFIGURATIONS
// ============================================

export const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.05,
    }
  },
};

export const staggerContainerFast = {
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.02,
    }
  },
};

// ============================================
// MODAL/OVERLAY VARIANTS
// ============================================

export const modalBackdrop = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: 0.3,
      ease: easings.signature,
    }
  },
  exit: {
    opacity: 0,
    transition: {
      duration: 0.2,
      ease: easings.exit,
    }
  },
};

export const modalContent = {
  initial: {
    opacity: 0,
    scale: 0.95,
    y: 20,
  },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      ...springs.smooth,
      duration: 0.5,
    }
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    y: 20,
    transition: {
      duration: 0.2,
      ease: easings.exit,
    }
  },
};

// ============================================
// SCROLL-BASED ANIMATIONS
// ============================================

export const scrollReveal = {
  initial: {
    opacity: 0,
    y: 30
  },
  whileInView: {
    opacity: 1,
    y: 0
  },
  transition: {
    duration: 0.6,
    ease: easings.emphasis,
  },
  viewport: {
    once: true,
    margin: "-50px",
  },
};

export const scrollRevealStagger = (index = 0) => ({
  initial: {
    opacity: 0,
    y: 30
  },
  whileInView: {
    opacity: 1,
    y: 0
  },
  transition: {
    duration: 0.6,
    ease: easings.emphasis,
    delay: index * 0.1,
  },
  viewport: {
    once: true,
    margin: "-50px",
  },
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Creates a staggered fade-in animation for list items
 * @param {number} index - Item index in list
 * @param {number} staggerDelay - Delay between items (default 0.1s)
 * @returns {object} Motion props
 */
export const staggerItem = (index, staggerDelay = 0.1) => ({
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: {
    duration: 0.6,
    ease: easings.emphasis,
    delay: index * staggerDelay,
  },
});

/**
 * Creates a glass morphism hover effect with backdrop blur
 * @returns {object} Motion props for hover state
 */
export const glassHover = {
  whileHover: {
    backdropFilter: "blur(16px)",
    backgroundColor: "rgba(255, 255, 255, 0.12)",
    transition: {
      duration: 0.4,
      ease: easings.hover,
    },
  },
};

/**
 * Detect if user prefers reduced motion
 * @returns {boolean}
 */
export const prefersReducedMotion = () => {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

/**
 * Get appropriate transition based on motion preferences
 * @param {object} normalTransition - Transition for normal motion
 * @param {object} reducedTransition - Transition for reduced motion (optional)
 * @returns {object}
 */
export const getTransition = (normalTransition, reducedTransition = { duration: 0.01 }) => {
  return prefersReducedMotion() ? reducedTransition : normalTransition;
};
