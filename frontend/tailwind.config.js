/** @type {import('tailwindcss').Config} */

/* Token names, radii, spacing and the type scale come from the design export,
 * which is not part of this repository. This config and src/styles/tokens.css
 * are together the authoritative definition — edit them directly.
 *
 * Colours resolve through CSS variables defined in src/styles/tokens.css so
 * one set of class names serves both themes. The design is natively dark; the
 * light palette was derived from it.
 */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'accent': 'rgb(var(--color-accent) / <alpha-value>)',
        'background': 'rgb(var(--color-background) / <alpha-value>)',
        'border': 'rgb(var(--color-border) / <alpha-value>)',
        'cta': 'rgb(var(--color-cta) / <alpha-value>)',
        'cta-hover': 'rgb(var(--color-cta-hover) / <alpha-value>)',
        'error': 'rgb(var(--color-error) / <alpha-value>)',
        'error-container': 'rgb(var(--color-error-container) / <alpha-value>)',
        'inverse-on-surface': 'rgb(var(--color-inverse-on-surface) / <alpha-value>)',
        'inverse-primary': 'rgb(var(--color-inverse-primary) / <alpha-value>)',
        'inverse-surface': 'rgb(var(--color-inverse-surface) / <alpha-value>)',
        'on-background': 'rgb(var(--color-on-background) / <alpha-value>)',
        'on-cta': 'rgb(var(--color-on-cta) / <alpha-value>)',
        'on-error': 'rgb(var(--color-on-error) / <alpha-value>)',
        'on-error-container': 'rgb(var(--color-on-error-container) / <alpha-value>)',
        'on-primary': 'rgb(var(--color-on-primary) / <alpha-value>)',
        'on-primary-container': 'rgb(var(--color-on-primary-container) / <alpha-value>)',
        'on-primary-fixed': 'rgb(var(--color-on-primary-fixed) / <alpha-value>)',
        'on-primary-fixed-variant': 'rgb(var(--color-on-primary-fixed-variant) / <alpha-value>)',
        'on-secondary': 'rgb(var(--color-on-secondary) / <alpha-value>)',
        'on-secondary-container': 'rgb(var(--color-on-secondary-container) / <alpha-value>)',
        'on-secondary-fixed': 'rgb(var(--color-on-secondary-fixed) / <alpha-value>)',
        'on-secondary-fixed-variant': 'rgb(var(--color-on-secondary-fixed-variant) / <alpha-value>)',
        'on-surface': 'rgb(var(--color-on-surface) / <alpha-value>)',
        'on-surface-variant': 'rgb(var(--color-on-surface-variant) / <alpha-value>)',
        'on-tertiary': 'rgb(var(--color-on-tertiary) / <alpha-value>)',
        'on-tertiary-container': 'rgb(var(--color-on-tertiary-container) / <alpha-value>)',
        'on-tertiary-fixed': 'rgb(var(--color-on-tertiary-fixed) / <alpha-value>)',
        'on-tertiary-fixed-variant': 'rgb(var(--color-on-tertiary-fixed-variant) / <alpha-value>)',
        'outline': 'rgb(var(--color-outline) / <alpha-value>)',
        'outline-variant': 'rgb(var(--color-outline-variant) / <alpha-value>)',
        'primary': 'rgb(var(--color-primary) / <alpha-value>)',
        'primary-container': 'rgb(var(--color-primary-container) / <alpha-value>)',
        'primary-fixed': 'rgb(var(--color-primary-fixed) / <alpha-value>)',
        'primary-fixed-dim': 'rgb(var(--color-primary-fixed-dim) / <alpha-value>)',
        'processing': 'rgb(var(--color-processing) / <alpha-value>)',
        'secondary': 'rgb(var(--color-secondary) / <alpha-value>)',
        'secondary-container': 'rgb(var(--color-secondary-container) / <alpha-value>)',
        'secondary-fixed': 'rgb(var(--color-secondary-fixed) / <alpha-value>)',
        'secondary-fixed-dim': 'rgb(var(--color-secondary-fixed-dim) / <alpha-value>)',
        'speaker-1': 'rgb(var(--color-speaker-1) / <alpha-value>)',
        'speaker-10': 'rgb(var(--color-speaker-10) / <alpha-value>)',
        'speaker-2': 'rgb(var(--color-speaker-2) / <alpha-value>)',
        'speaker-3': 'rgb(var(--color-speaker-3) / <alpha-value>)',
        'speaker-4': 'rgb(var(--color-speaker-4) / <alpha-value>)',
        'speaker-5': 'rgb(var(--color-speaker-5) / <alpha-value>)',
        'speaker-6': 'rgb(var(--color-speaker-6) / <alpha-value>)',
        'speaker-7': 'rgb(var(--color-speaker-7) / <alpha-value>)',
        'speaker-8': 'rgb(var(--color-speaker-8) / <alpha-value>)',
        'speaker-9': 'rgb(var(--color-speaker-9) / <alpha-value>)',
        'success': 'rgb(var(--color-success) / <alpha-value>)',
        'surface': 'rgb(var(--color-surface) / <alpha-value>)',
        'surface-bright': 'rgb(var(--color-surface-bright) / <alpha-value>)',
        'surface-container': 'rgb(var(--color-surface-container) / <alpha-value>)',
        'surface-container-high': 'rgb(var(--color-surface-container-high) / <alpha-value>)',
        'surface-container-highest': 'rgb(var(--color-surface-container-highest) / <alpha-value>)',
        'surface-container-low': 'rgb(var(--color-surface-container-low) / <alpha-value>)',
        'surface-container-lowest': 'rgb(var(--color-surface-container-lowest) / <alpha-value>)',
        'surface-dim': 'rgb(var(--color-surface-dim) / <alpha-value>)',
        'surface-raised': 'rgb(var(--color-surface-raised) / <alpha-value>)',
        'surface-tint': 'rgb(var(--color-surface-tint) / <alpha-value>)',
        'surface-variant': 'rgb(var(--color-surface-variant) / <alpha-value>)',
        'tertiary': 'rgb(var(--color-tertiary) / <alpha-value>)',
        'tertiary-container': 'rgb(var(--color-tertiary-container) / <alpha-value>)',
        'tertiary-fixed': 'rgb(var(--color-tertiary-fixed) / <alpha-value>)',
        'tertiary-fixed-dim': 'rgb(var(--color-tertiary-fixed-dim) / <alpha-value>)',
        'text-faint': 'rgb(var(--color-text-faint) / <alpha-value>)',
        'text-muted': 'rgb(var(--color-text-muted) / <alpha-value>)',
        'text-primary': 'rgb(var(--color-text-primary) / <alpha-value>)',
      },
      borderRadius: {
          "DEFAULT": "0.25rem",
          "lg": "0.5rem",
          "xl": "0.75rem",
          "full": "9999px"
        },
      spacing: {
          "margin-page": "2rem",
          "sidebar-width": "260px",
          "sidebar-collapsed": "76px",
          "max-content-width": "768px",
          "gutter": "1.5rem"
        },
      fontFamily: {
          "transcript-body": [
            "Inter"
          ],
          "meta-data": [
            "Inter"
          ],
          "headline-lg-mobile": [
            "Plus Jakarta Sans"
          ],
          "sidebar-header": [
            "Plus Jakarta Sans"
          ],
          "transcript-body-hi": [
            "Noto Sans Devanagari"
          ],
          "label-sm": [
            "Inter"
          ],
          "headline-lg": [
            "Plus Jakarta Sans"
          ]
        },
      fontSize: {
          /* Used by the mobile bottom tab bar in the export, but absent from
             every exported config — so it rendered at inherited size. One step
             below label-sm, same weight and tracking. */
          "label-sm-mobile": [
            "10px",
            {
              "lineHeight": "14px",
              "letterSpacing": "0.05em",
              "fontWeight": "600"
            }
          ],
          "transcript-body": [
            "12px",
            {
              "lineHeight": "1.5",
              "fontWeight": "400"
            }
          ],
          "meta-data": [
            "11px",
            {
              "lineHeight": "15px",
              "fontWeight": "400"
            }
          ],
          "headline-lg-mobile": [
            "17px",
            {
              "lineHeight": "24px",
              "fontWeight": "600"
            }
          ],
          "sidebar-header": [
            "15px",
            {
              "lineHeight": "20px",
              "fontWeight": "600"
            }
          ],
          "transcript-body-hi": [
            "12px",
            {
              "lineHeight": "1.5",
              "fontWeight": "400"
            }
          ],
          "label-sm": [
            "11px",
            {
              "lineHeight": "16px",
              "letterSpacing": "0.05em",
              "fontWeight": "600"
            }
          ],
          "headline-lg": [
            "30px",
            {
              "lineHeight": "38px",
              "letterSpacing": "-0.02em",
              "fontWeight": "600"
            }
          ]
        },
    },
  },
  plugins: [],
}
