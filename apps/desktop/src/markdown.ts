import remarkGfm from 'remark-gfm';

// No raw HTML plugin or custom URL transform: keep react-markdown's safe URL
// handling while supporting report tables, task lists and strikethrough.
export const markdownOptions = { remarkPlugins: [remarkGfm], skipHtml: true };
