# Accessibility Guide

## Tools Installed

### ✅ eslint-plugin-jsx-a11y
Static analysis tool that checks JSX for accessibility issues during development.

**Run linting:**
```bash
npm run lint
```

This will catch issues like:
- Missing alt text on images
- Missing ARIA labels
- Invalid ARIA attributes
- Missing keyboard event handlers
- Semantic HTML issues

## Accessibility Features Implemented

### ✅ Keyboard Navigation
- **Ctrl/Cmd + Enter** to generate comments (keyboard shortcut)
- All interactive elements are keyboard accessible
- Focus indicators on all focusable elements

### ✅ ARIA Labels & Roles
- `role="alert"` on error messages
- `aria-live="assertive"` for error announcements
- `aria-label` on buttons and interactive elements
- `aria-busy` on loading states
- `aria-labelledby` for associating labels with content
- `aria-describedby` for input help text

### ✅ Semantic HTML
- `<main>` for main content
- `<header>` for page header
- `<section>` for major sections
- `<article>` for algorithm results
- `<aside>` for supplementary information (tokens)
- `<footer>` for metadata

### ✅ Screen Reader Support
- `.sr-only` class for screen-reader-only text
- Descriptive labels for all interactive elements
- Runtime and status information announced to screen readers

### ✅ Visual Accessibility
- **Open Sans font** - accessible, highly readable typeface
- Proper color contrast (checked with Tailwind defaults)
- Focus rings on all interactive elements
- Loading states clearly indicated
- Error states clearly indicated

### ✅ Form Accessibility
- `<label>` properly associated with `<textarea>` via `htmlFor` and `id`
- `aria-describedby` linking help text to input
- Placeholder text provides guidance

## External Tools

### Chrome Extension: Accessibility Insights
1. Install from: https://accessibilityinsights.io/docs/web/overview/
2. Use it to test:
   - Color contrast ratios
   - Keyboard navigation
   - Screen reader compatibility
   - ARIA implementation

### Manual Testing Checklist
- [ ] Tab through all interactive elements
- [ ] Test with screen reader (NVDA/JAWS/VoiceOver)
- [ ] Verify color contrast (WCAG AA minimum)
- [ ] Test keyboard shortcuts (Ctrl+Enter)
- [ ] Verify focus indicators visible
- [ ] Test with browser zoom (200%)

## WCAG 2.1 Compliance

### Level A (Required)
- ✅ Keyboard accessible
- ✅ No keyboard traps
- ✅ Proper labeling
- ✅ Error identification

### Level AA (Recommended)
- ✅ Color contrast (4.5:1 for text)
- ✅ Text resizable to 200%
- ✅ Focus indicators
- ✅ Consistent navigation

## Running Accessibility Tests

```bash
# Lint for accessibility issues
npm run lint

# Check for specific rules
npx eslint --ext .tsx --config eslint.config.mjs src/
```

## Resources

- [WCAG 2.1 Guidelines](https://www.w3.org/TR/WCAG21/)
- [React Accessibility Docs](https://legacy.reactjs.org/docs/accessibility.html)
- [eslint-plugin-jsx-a11y Rules](https://github.com/jsx-eslint/eslint-plugin-jsx-a11y#supported-rules)
- [Accessibility Insights](https://accessibilityinsights.io/docs/web/overview/)

