/**
 * Material Symbols Outlined, the icon set the Stitch export uses.
 *
 * The font is self-hosted (see src/styles/index.css) rather than pulled from
 * the Google Fonts CDN, so it works with no network at the stand.
 *
 * `filled` maps to the FILL axis of the variable font. The export uses it for
 * the active state of a nav item and for the brand mark, which is why it is a
 * prop rather than a separate icon name.
 */
export default function Icon({ name, filled = false, className = '', size, style, ...rest }) {
  return (
    <span
      aria-hidden="true"
      className={`material-symbols-outlined select-none ${className}`}
      style={{
        ...(filled ? { fontVariationSettings: "'FILL' 1" } : null),
        ...(size ? { fontSize: `${size}px` } : null),
        ...style,
      }}
      {...rest}
    >
      {name}
    </span>
  )
}
