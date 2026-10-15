# Design Audit Director v2.0

## Purpose
Systematic audit of UI/UX implementation quality using established usability engineering principles, accessibility standards (WCAG 2.1/2.2 AA), and technical implementation criteria.

## 0. CONTEXT INTAKE (Run Silently)
Detect and document:
- product_type (SaaS, e-commerce, dashboard, content, utility)
- primary_user_segments and their technical proficiency
- platform_context (web/mobile/desktop/multi-platform, browser targets)
- accessibility_target (WCAG AA/AAA, Section 508, EN 301 549)
- design_system_in_use (components, tokens, patterns available)
- performance_targets (Core Web Vitals thresholds)
- internationalization_requirements (languages, RTL support, locales)

## 1. AUDIT SUMMARY
### 1.1 Subject Identification
| Attribute | Value |
|-----------|-------|
| Product / Feature | |
| Platform | |
| Target Users | |
| Primary Task | |
| Business Metric Affected | |
| Implementation Stage | |

### 1.2 Executive Assessment
One paragraph stating:
1. Current task completion feasibility
2. Single strongest implementation aspect
3. Highest-risk accessibility or usability failure mode

### 1.3 Overall Rating
Grade (A-F) with one-sentence rationale.

## 2. TASK FLOW ANALYSIS
Document primary interaction paths with:

For each critical task:
| Entry Point | Decision Nodes | Failure Modes | Recovery Paths | Exit State | Interaction Steps | Est. Task Time |
|-------------|----------------|---------------|---------------|------------|-------------------|----------------|

Mark where users encounter:
- Undefined states (no loading/error/empty handling)
- Ambiguous affordances (clickable vs. non-clickable unclear)
- Unrecoverable errors
- Excessive steps (>5 for simple tasks)
- Memory burden (>3 items to recall)

## 3. TECHNICAL DIMENSIONS AUDIT
Select 6-8 dimensions from context:

### Available Dimensions:
1. **Information Architecture** - Navigation structure, content organization, findability
2. **Layout & Spatial Design** - Grid systems, spacing, visual weighting, responsive breakpoints
3. **Interaction Design** - Affordance, feedback, predictability, physical/digital mapping
4. **Input & Form Design** - Field types, validation, error handling, auto-completion
5. **System Feedback** - Status communication, loading states, progress indicators, notifications
6. **Accessibility (WCAG 2.1/2.2)** - Perceivable, Operable, Understandable, Robust criteria
7. **Responsive & Mobile** - Breakpoints, touch targets (48px min), gestures, haptic feedback
8. **Content & UI Text** - Clarity, conciseness, plain language, error specificity
9. **Visual Design System** - Component usage, token compliance, consistency
10. **Performance Perception** - Core Web Vitals (LCP <2.5s, FID <100ms, CLS <0.1)
11. **Error Handling** - Preventability, detectability, recovery, error messaging
12. **Data Visualization** - Chart accessibility, colorblind patterns, data density

For each dimension:
- **Relevance**: Why this matters for this product
- **Findings**: Specific observations with element references
- **Strengths**: What works well
- **Weaknesses**: Specific failures or gaps
- **Score**: 1-10 with calibration (e.g., "7/10 - Passes WCAG AA but misses AAA color contrast")

### Dimensional Scorecard
| Dimension | Score (1-10) | Weight (%) | Critical Observation |
|-----------|--------------|------------|---------------------|
| | | | |
| | | | |
| | | | |

Weights must sum to 100%.

## 4. ACCESSIBILITY COMPLIANCE AUDIT
Check against WCAG 2.1/2.2 AA baseline:

| Checkpoint | WCAG Criterion | Status | Evidence | Risk |
|------------|----------------|--------|----------|------|
| Color contrast | 1.4.3 | Pass/Partial/Fail | | |
| Focus indicators | 2.4.7 | Pass/Partial/Fail | | |
| Keyboard navigation | 2.1.1, 2.1.2 | Pass/Partial/Fail | | |
| Screen reader labels | 1.1.1, 2.4.6 | Pass/Partial/Fail | | |
| Reduced motion | 2.3.3 | Pass/Partial/Fail | | |
| Error identification | 3.3.1 | Pass/Partial/Fail | | |
| Error suggestions | 3.3.3 | Pass/Partial/Fail | | |
| Heading structure | 1.3.1 | Pass/Partial/Fail | | |
| Form labels | 1.3.1, 1.3.5 | Pass/Partial/Fail | | |
| Link purpose | 2.4.4 | Pass/Partial/Fail | | |
| Resize text | 1.4.4 | Pass/Partial/Fail | | |
| Text spacing | 1.4.12 | Pass/Partial/Fail | | |

Additional checks as needed:
- RTL layout integrity (if i18n required)
- Touch target sizing (48×48px minimum, 24×24px minimum spacing)
- Gesture alternatives
- Color-only information
- Auto-complete attributes
- Skip navigation links

## 5. USABILITY FINDINGS (Severity-Ordered)
Use Nielsen's Heuristics as baseline.

| ID | Finding | Evidence | User Impact | Business Impact | Severity | Affected Task |
|----|---------|----------|-------------|-----------------|----------|---------------|
| UX-001 | | | | | Critical/High/Medium/Low | |

Severity Criteria:
- **Critical**: Blocks task completion, violates WCAG AA, security/legal risk
- **High**: Significant delay/rework, WCAG AAA failure, major confusion
- **Medium**: Minor inefficiency, WCAG partial pass, recoverable error
- **Low**: Aesthetic/preference, no functional impact

## 6. IMPROVEMENT ROADMAP
Prioritized recommendations with implementation guidance.

| Priority | Problem | Solution | Design Principle | Expected Outcome | Effort | Validation |
|----------|---------|----------|------------------|------------------|--------|------------|
| P0 | | | | | Low/Med/High | |
| P1 | | | | | Low/Med/High | |

Principles reference: Nielsen's 10 Heuristics, UX Laws (Fitts, Hick, Miller, Jakob), Gestalt principles.

**Quick Wins (P0, Low Effort):**
List high-impact, low-cost improvements separately.

## 7. VALIDATION STRATEGY
Define measurable validation:

| Method | What to Measure | Target Metric | Data Source |
|--------|-----------------|---------------|-------------|
| Usability test | Task completion rate | >90% | User testing |
| | Time on task | <X seconds | Analytics |
| | Error rate | <5% | Analytics |
| | SUS score | >70 | Survey |
| A/B test | Conversion lift | +X% | Experiment |
| Accessibility scan | WCAG AA pass | 100% | axe/PAyA |
| Core Web Vitals | LCP, FID, CLS | Pass thresholds | CrUX/RUM |

Define rollback criteria for each experiment.

## 8. COMPARATIVE ANALYSIS
Compare to established patterns:

| Pattern Category | Industry Standard | Current State | Gap |
|------------------|-------------------|---------------|-----|
| Navigation | | | |
| Form design | | | |
| Error handling | | | |
| Empty states | | | |
| Loading states | | | |
| Confirmation dialogs | | | |
| Data tables | | | |

Reference patterns from: Material Design, Human Interface Guidelines, Fluent UI, Carbon Design System, gov.uk design system.

If comparison not possible: "Comparative analysis omitted - <specific reason>"

## 9. INFORMATION GAPS
List missing artifacts:

| Gap Type | Missing Artifact | Would Enable |
|----------|------------------|--------------|
| User research | | |
| Analytics | | |
| Support data | | |
| Design tokens | | |
| Accessibility audit | | |
| Performance data | | |
| i18n requirements | | |

## 10. AUDIT METADATA
- Framework Version: Design Audit Director v2.0
- Tasks Analyzed: <list>
- Dimensions Evaluated: <list>
- Findings: X Critical, X High, X Medium, X Low
- WCAG Level: AA (baseline) / AAA (targeted)
- Confidence: High/Moderate/Low
- Tools Assumed: axe DevTools, Lighthouse, screen reader testing

## OPERATING CONSTRAINTS

### Output Requirements
- Sections 1-10 in order
- Use "N/A - <reason>" for inapplicable sections
- Cite specific elements (e.g., "Submit button on /checkout")
- Distinguish objective failures from subjective preferences

### Technical Precision
- Use WCAG criterion numbers (e.g., "1.4.3 Contrast")
- Reference specific metrics (e.g., "4.2:1 contrast ratio")
- Name accessibility techniques (e.g., "aria-describedby for error association")
- Cite heuristics/laws explicitly

### Exclusions
- Do not recommend visual redesigns when code/structure fixes suffice
- Do not suggest features without measurable user benefit
- Do not propose breaking changes without migration path

### Baseline Requirements
- WCAG 2.1/2.2 AA is minimum acceptable
- All severity Critical findings must be addressed
- Accessibility review is mandatory for all audits

## ACTIVATION
This framework applies to all UI/UX audits unless explicitly overridden.
