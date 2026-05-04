---
name: The Bard's Interface
colors:
  surface: '#fbfbe2'
  surface-dim: '#dbdcc3'
  surface-bright: '#fbfbe2'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f5dc'
  surface-container: '#efefd7'
  surface-container-high: '#eaead1'
  surface-container-highest: '#e4e4cc'
  on-surface: '#1b1d0e'
  on-surface-variant: '#5a403c'
  inverse-surface: '#303221'
  inverse-on-surface: '#f2f2d9'
  outline: '#8e706b'
  outline-variant: '#e3beb8'
  surface-tint: '#b52619'
  primary: '#610000'
  on-primary: '#ffffff'
  primary-container: '#8b0000'
  on-primary-container: '#ff907f'
  inverse-primary: '#ffb4a8'
  secondary: '#735c00'
  on-secondary: '#ffffff'
  secondary-container: '#fed65b'
  on-secondary-container: '#745c00'
  tertiary: '#2c2c2c'
  on-tertiary: '#ffffff'
  tertiary-container: '#434242'
  on-tertiary-container: '#b1aeae'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdad4'
  primary-fixed-dim: '#ffb4a8'
  on-primary-fixed: '#410000'
  on-primary-fixed-variant: '#920703'
  secondary-fixed: '#ffe088'
  secondary-fixed-dim: '#e9c349'
  on-secondary-fixed: '#241a00'
  on-secondary-fixed-variant: '#574500'
  tertiary-fixed: '#e5e2e1'
  tertiary-fixed-dim: '#c8c6c5'
  on-tertiary-fixed: '#1c1b1b'
  on-tertiary-fixed-variant: '#474746'
  background: '#fbfbe2'
  on-background: '#1b1d0e'
  surface-variant: '#e4e4cc'
typography:
  headline-lg:
    fontFamily: Newsreader
    fontSize: 40px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Newsreader
    fontSize: 32px
    fontWeight: '500'
    lineHeight: '1.3'
  headline-sm:
    fontFamily: Newsreader
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.4'
  body-lg:
    fontFamily: Noto Serif
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Noto Serif
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  label-md:
    fontFamily: Noto Serif
    fontSize: 14px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
  gutter: 24px
  margin: 32px
---

## Brand & Style

The brand personality of the design system is intellectual, poetic, and authoritative, yet possesses a modern clarity that ensures seamless digital interaction. It is designed for a target audience that appreciates literary depth and classical aesthetics without sacrificing the speed and efficiency of contemporary AI.

The design style is a sophisticated blend of **Tactile Skeuomorphism** and **Minimalism**. It leverages physical metaphors—such as the texture of hand-pressed paper and the weight of ink—while maintaining a clean, structured layout. The emotional response should be one of "discovery in a library," evoking the quiet focus and timelessness of an Elizabethan study. Visual elements are intentional, using fine-line borders and classical motifs like laurel wreaths to guide the eye rather than distract it.

## Colors

The color palette is grounded in the materiality of the late 16th century. The foundation is **Parchment White**, a warm, non-reflective neutral that reduces eye strain and provides a historic backdrop. **Ink Black** is used for all primary communication to ensure maximum legibility and to mimic the appearance of iron-gall ink.

**Deep Crimson** serves as the primary action color, used for high-priority buttons and critical states, evoking the wax seals used on royal correspondence. **Weathered Gold** acts as the highlight color, reserved for interactive states, focus rings, and decorative accents that signify premium quality or "enlightened" insights from the chatbot.

## Typography

This design system utilizes a dual-serif approach to reinforce its literary character. For headlines, **Newsreader** provides an authoritative and editorial tone with its slightly condensed, classic proportions. It should be used for titles, headers, and major chatbot statements to command attention.

For body text and functional labels, **Noto Serif** offers exceptional legibility in a digital environment while maintaining a timeless, premium feel. Body text should be set with generous line heights to mimic the airy layout of a well-typeset folio. Labels use a slight letter-spacing increase and uppercase styling to differentiate functional metadata from narrative content.

## Layout & Spacing

The design system employs a **Fixed Grid** model for the primary chat interface to evoke the structured margins of a manuscript. The content is centered within a 960px container, flanked by generous safe margins that allow the background parchment texture to breathe.

The spacing rhythm is based on a 4px baseline, but defaults to larger increments (16px and 24px) to create a sense of classical elegance and prevent the UI from feeling "crowded." Padding within components like chat bubbles should be substantial, ensuring that text never feels cramped against its container edges, much like the wide margins of a luxury 16th-century print.

## Elevation & Depth

Depth in this design system is conveyed through **Tonal Layers** and texture rather than modern drop shadows. Surfaces are stacked to look like sheets of paper resting atop one another. 

To achieve this, use:
- **Paper Texturing:** A subtle, low-opacity grainy overlay on all Parchment White surfaces.
- **Fine-Line Borders:** 1px solid borders in a slightly darker parchment shade (#E8E8C0) or Ink Black to define containers.
- **Stacked Surfaces:** Secondary panels (like a history sidebar) should appear "underneath" the main chat area by using a slightly darker or more textured neutral tone.
- **Minimal Inner Glows:** Use very soft, dark inner glows on input fields to suggest a slight indentation in the "paper."

## Shapes

The shape language is primarily **Soft (0.25rem)**, reflecting the organic, slightly irregular nature of hand-cut parchment. Avoid perfect circles or high-radius pills, as they feel too modern and "plastic." 

Buttons and chat bubbles use the base roundedness to take the "edge" off the containers, while large cards and the main chat container should remain sharp or use minimal 2px rounding to maintain the structural integrity of a book or folio page.

## Components

### Buttons
Buttons should feature a calligraphic feel. Use the **Deep Crimson** for primary actions with **Weathered Gold** text. The border should be a double-line (a 1px solid line inside a 2px offset container) to mimic formal document headings. Hover states should transition the gold text to a brighter shine.

### Chat Bubbles
Chat bubbles must resemble "snippets" of parchment. They should not use heavy rounded corners. Instead, use a very subtle deckle-edge effect (irregular border-radius or a mask) on one side. The chatbot's bubbles are **Parchment White** with a fine **Ink Black** border, while the user's bubbles use a faint **Weathered Gold** tint to distinguish the dialogue.

### Input Fields
The text input area should resemble a simple line on a page. Use a 1px **Ink Black** bottom border only. When focused, a small **Quill Icon** should appear at the start of the line, and the bottom border should transition to **Weathered Gold**.

### Decorative Motifs
Use a **Laurel Wreath** icon to denote "Success" states or completed tasks. Use a **Quill** icon as the "Send" button. Fine horizontal lines with a small diamond or flourish in the center should be used to separate different days in the chat history.

### Lists and Menus
Menus should appear as "Overlays" with a distinct paper texture and a 1px **Weathered Gold** border, feeling like a small pamphlet or note placed over the main interface.