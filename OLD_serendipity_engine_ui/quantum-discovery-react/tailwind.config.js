/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'venture-burgundy': '#7d1935',
        'venture-dark': '#5a1227',
        'venture-card': '#8b2346',
        'venture-border': '#a73959',
        'venture-light': '#fde8eb',
        'venture-text': '#f5f5f5',
        'venture-accent': '#34d399',
        'venture-green': '#10b981',
        'venture-score': '#92c5c5',
      },
      animation: {
        'pulse-scale': 'pulseScale 1s infinite',
        'slide-in': 'slideIn 0.3s ease-out',
      },
      keyframes: {
        pulseScale: {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
        },
        slideIn: {
          '0%': { 
            opacity: '0',
            transform: 'translateY(20px)'
          },
          '100%': { 
            opacity: '1',
            transform: 'translateY(0)'
          },
        },
      },
    },
  },
  plugins: [],
}