//! SVG & LaTeX/TikZ Architecture Diagram Exporters (`diagram_export`).
//!
//! Renders any `AiDesignSpec` computation graph as a standalone vector `.svg` diagram
//! or publication-ready `\begin{tikzpicture}...\end{tikzpicture}` LaTeX figure,
//! bounding visual detail cleanly for large graphs (1,000+ nodes).

use crate::analysis::infer_shapes;
use crate::spec::AiDesignSpec;
use std::fmt::Write;

const MAX_RENDERED_SVG_NODES: usize = 128;

/// Exports an `AiDesignSpec` architecture diagram as a standalone vector SVG string.
pub fn export_spec_to_svg(spec: &AiDesignSpec) -> String {
    let shape_report = infer_shapes(spec);
    let total_nodes = spec.graph.nodes.len();
    let render_count = total_nodes.min(MAX_RENDERED_SVG_NODES);
    let truncated = total_nodes.saturating_sub(render_count);

    let card_w = 240usize;
    let card_h = 68usize;
    let gap_x = 40usize;
    let pad_x = 32usize;
    let item_count = spec.inputs.len() + render_count + usize::from(truncated > 0) + spec.outputs.len();
    let total_w = (pad_x * 2 + item_count * (card_w + gap_x)).max(420);
    let total_h = 180usize;

    let mut svg = String::with_capacity(item_count * 280 + 512);
    let _ = writeln!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">"##
    );
    let _ = writeln!(
        svg,
        r##"  <rect width="100%" height="100%" fill="#0f172a"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <text x="32" y="28" fill="#e2e8f0" font-family="monospace" font-size="14" font-weight="bold">{} ({} nodes)</text>"##,
        escape_xml(&spec.metadata.display_name),
        total_nodes
    );

    let y = 64usize;
    let mut x = pad_x;
    let mut prev_right: Option<usize> = None;

    let draw_box = |buf: &mut String,
                    bx: usize,
                    by: usize,
                    stroke: &str,
                    title: &str,
                    sub: &str,
                    prev: &mut Option<usize>| {
        if let Some(px) = *prev {
            let cy = by + card_h / 2;
            let _ = writeln!(
                buf,
                r##"  <line x1="{px}" y1="{cy}" x2="{bx}" y2="{cy}" stroke="#64748b" stroke-width="2"/>"##
            );
        }
        let _ = writeln!(
            buf,
            r##"  <rect x="{bx}" y="{by}" width="{card_w}" height="{card_h}" rx="8" fill="#1e293b" stroke="{stroke}" stroke-width="2"/>"##
        );
        let _ = writeln!(
            buf,
            r##"  <text x="{}" y="{}" fill="#f8fafc" font-family="monospace" font-size="12" font-weight="bold">{}</text>"##,
            bx + 12,
            by + 26,
            escape_xml(title)
        );
        let _ = writeln!(
            buf,
            r##"  <text x="{}" y="{}" fill="#94a3b8" font-family="monospace" font-size="11">{}</text>"##,
            bx + 12,
            by + 48,
            escape_xml(sub)
        );
        *prev = Some(bx + card_w);
    };

    for inp in &spec.inputs {
        let shape = shape_report
            .input_shapes
            .get(&inp.id)
            .map(|s| s.display())
            .unwrap_or_else(|| "[-]".to_string());
        draw_box(
            &mut svg,
            x,
            y,
            "#38bdf8",
            &format!("Input: {}", inp.id),
            &shape,
            &mut prev_right,
        );
        x += card_w + gap_x;
    }

    for node in spec.graph.nodes.iter().take(render_count) {
        let out_shape = shape_report
            .node_shapes
            .get(&node.id)
            .and_then(|n| n.output_shape.as_ref())
            .map(|s| s.display())
            .unwrap_or_else(|| node.operator.category_tag().to_string());
        let prec = node
            .precision
            .unwrap_or(spec.precision.default_weight_precision)
            .label();
        draw_box(
            &mut svg,
            x,
            y,
            "#a78bfa",
            &node.id,
            &format!("{} [{}]", out_shape, prec),
            &mut prev_right,
        );
        x += card_w + gap_x;
    }

    if truncated > 0 {
        draw_box(
            &mut svg,
            x,
            y,
            "#f59e0b",
            &format!("... +{truncated} more nodes"),
            "LOD summary",
            &mut prev_right,
        );
        x += card_w + gap_x;
    }

    for out in &spec.outputs {
        let shape = shape_report
            .output_shapes
            .get(&out.id)
            .map(|s| s.display())
            .unwrap_or_else(|| "[-]".to_string());
        draw_box(
            &mut svg,
            x,
            y,
            "#34d399",
            &format!("Output: {}", out.id),
            &shape,
            &mut prev_right,
        );
        x += card_w + gap_x;
    }

    svg.push_str("</svg>\n");
    svg
}

/// Exports an `AiDesignSpec` architecture diagram as a standalone LaTeX TikZ figure.
pub fn export_spec_to_tikz(spec: &AiDesignSpec) -> String {
    let total_nodes = spec.graph.nodes.len();
    let render_count = total_nodes.min(MAX_RENDERED_SVG_NODES);
    let truncated = total_nodes.saturating_sub(render_count);

    let mut tikz = String::with_capacity(render_count * 120 + 256);
    tikz.push_str("% Auto-generated by AI Designer Studio\n");
    tikz.push_str("\\begin{tikzpicture}[node distance=1.4cm, auto,\n");
    tikz.push_str("  block/.style={rectangle, draw, rounded corners, minimum height=2.2em, minimum width=7em, font=\\small\\ttfamily}]\n");

    let mut prev_name: Option<String> = None;
    for (idx, inp) in spec.inputs.iter().enumerate() {
        let node_key = format!("in_{idx}");
        let pos = prev_name
            .as_ref()
            .map(|p| format!(", right=of {p}"))
            .unwrap_or_default();
        let _ = writeln!(
            tikz,
            "  \\node[block{pos}] ({node_key}) {{Input: {}}};",
            escape_tex(&inp.id)
        );
        if let Some(p) = &prev_name {
            let _ = writeln!(tikz, "  \\draw[->] ({p}) -- ({node_key});");
        }
        prev_name = Some(node_key);
    }

    for (idx, node) in spec.graph.nodes.iter().take(render_count).enumerate() {
        let node_key = format!("layer_{idx}");
        let pos = prev_name
            .as_ref()
            .map(|p| format!(", right=of {p}"))
            .unwrap_or_default();
        let _ = writeln!(
            tikz,
            "  \\node[block{pos}] ({node_key}) {{{}}};",
            escape_tex(&node.id)
        );
        if let Some(p) = &prev_name {
            let _ = writeln!(tikz, "  \\draw[->] ({p}) -- ({node_key});");
        }
        prev_name = Some(node_key);
    }

    if truncated > 0 {
        let node_key = "ellipsis".to_string();
        let pos = prev_name
            .as_ref()
            .map(|p| format!(", right=of {p}"))
            .unwrap_or_default();
        let _ = writeln!(
            tikz,
            "  \\node[block{pos}] ({node_key}) {{+{truncated} nodes}};"
        );
        if let Some(p) = &prev_name {
            let _ = writeln!(tikz, "  \\draw[->] ({p}) -- ({node_key});");
        }
        prev_name = Some(node_key);
    }

    for (idx, out) in spec.outputs.iter().enumerate() {
        let node_key = format!("out_{idx}");
        let pos = prev_name
            .as_ref()
            .map(|p| format!(", right=of {p}"))
            .unwrap_or_default();
        let _ = writeln!(
            tikz,
            "  \\node[block{pos}] ({node_key}) {{Output: {}}};",
            escape_tex(&out.id)
        );
        if let Some(p) = &prev_name {
            let _ = writeln!(tikz, "  \\draw[->] ({p}) -- ({node_key});");
        }
        prev_name = Some(node_key);
    }

    tikz.push_str("\\end{tikzpicture}\n");
    tikz
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_tex(s: &str) -> String {
    s.replace('_', "\\_")
        .replace('&', "\\&")
        .replace('%', "\\%")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::roofline::analyze_roofline;
    use crate::spec::archetypes::{ArchitectureArchetype, scaffold_archetype};
    use crate::spec::{ConnectionSpec, NodeSpec, OperatorKind, PrecisionFormat};

    #[test]
    fn test_svg_and_tikz_diagram_export() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "diag_llm",
            "Diagram LLM",
            256,
            2,
            PrecisionFormat::BF16,
        );
        let svg = export_spec_to_svg(&spec);
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Diagram LLM"));

        let tikz = export_spec_to_tikz(&spec);
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("\\end{tikzpicture}"));
        assert!(tikz.contains("\\draw[->]"));
    }

    #[test]
    fn test_1000_node_graph_validation_shape_roofline_and_diagram_scale() {
        let mut spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "scale_1000",
            "1000-Node Stress Spec",
            256,
            1,
            PrecisionFormat::BF16,
        );
        spec.graph.nodes.clear();
        spec.graph.connections.clear();

        for i in 0..1000usize {
            spec.graph.nodes.push(NodeSpec::new(
                format!("layer_{i}"),
                format!("Layer {i}"),
                OperatorKind::Linear {
                    in_features: 256,
                    out_features: 256,
                    bias: false,
                },
            ));
            if i > 0 {
                spec.graph.connections.push(ConnectionSpec {
                    from_node: format!("layer_{}", i - 1),
                    from_port: 0,
                    to_node: format!("layer_{i}"),
                    to_port: 0,
                });
            }
        }

        spec.validate().expect("1000-node spec should validate");
        let shapes = infer_shapes(&spec);
        assert_eq!(shapes.node_shapes.len(), 1000);
        let roofline = analyze_roofline(&spec, 4, 128);
        assert!(roofline.total_flops_per_token > 0);
        let svg = export_spec_to_svg(&spec);
        assert!(svg.contains("1000 nodes"));
        assert!(svg.contains("LOD summary"));
        let tikz = export_spec_to_tikz(&spec);
        assert!(tikz.contains("nodes"));
    }
}
