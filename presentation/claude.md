Agent Instructions for Presentation & Frontend Sync
Note:
Claude, your objective is to implement this reveal.js presentation and synchronize it with the Next.js frontend. You must adopt a dual-purpose development strategy: any new visualizations required for this deck must be built as reusable React components in the Next.js frontend first, then embedded here via iframes. The deck itself must strictly use reveal.js Markdown plugin syntax.

--

Theme & Styling Rules
Palette: Use the same pallette as the Next.js frontend, however the background will be white and the text black in the presentation.

CSS Customization: Override default reveal.js headings (h1, h2, h3) to use the Orange accent.

Visibility: Ensure all code blocks and diagrams have high contrast for projector visibility.

Layout: Keep slides minimalist. Maximum 4 bullet points per slide. Use icons (e.g. boxicons) as the bullets where appropriate. Ensure formatting looks clean and modern and stays consistent throughout the presentation.

Note:
Claude, you will need to create a custom CSS file (e.g., theme/wcci-custom.css) within the reveal.js directory. Set the --r-background-color to white, --r-main-color to black, and --r-heading-color to our specific orange hex. Since projectors wash out dark backgrounds, the white theme is mandatory. Any embedded Next.js iframes should also accept a URL parameter (e.g., ?theme=light) to switch their internal Deck.gl maps to a light MapLibre basemap to match the slides.

--

Component Integration Strategy
Live Dashboards: Use <iframe data-src="/route" data-preload> for dynamic WebGL components.

Static Figures: If a live component is too heavy, build it in Next.js, screenshot it, and use standard Markdown image tags.

Architecture Diagrams: Use Mermaid.js natively in reveal.js or embed static draw.io SVG exports.

Note:
Claude, do not build isolated HTML/JS charts just for this presentation. If a slide calls for a graph of the "Null-Zero Conflation," build that chart as a route in the Next.js app (e.g., /figures/null-zero) and embed it. This ensures our digital twin frontend gains new analytical views while we build the defense deck, killing two birds with one stone.