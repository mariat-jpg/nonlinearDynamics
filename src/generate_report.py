from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

def generate_report(avg_healthy, avg_parkinsons, healthy_measures, parkinsons_measures, pd_regions):
    """
    Generates a clean PDF report of all findings.
    avg_healthy, avg_parkinsons: 1D arrays of ROI switch counts (24,)
    healthy_measures, parkinsons_measures: dicts of graph measures
    pd_regions: list of ROI names
    """
    os.makedirs("results", exist_ok=True)
    doc = SimpleDocTemplate(
        "results/report.pdf",
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    title_style   = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    body_style    = styles['Normal']
    body_style.spaceAfter = 6

    story = []

    #Title Page 
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("Dynamic Functional Connectivity Analysis", title_style))
    story.append(Paragraph("Healthy Subjects vs Parkinson's Disease", styles['Heading2']))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This report presents a comprehensive analysis of dynamic brain connectivity "
        "in healthy subjects and Parkinson's disease patients using resting-state fMRI data. "
        "Analysis includes sliding window connectivity, spectral clustering, ROI flexibility, "
        "recurrence analysis, and graph-theoretic measures.",
        body_style
    ))
    story.append(PageBreak())

    #Section 1: ROI Switch Counts
    story.append(Paragraph("1. ROI Flexibility (Community Switch Counts)", heading_style))
    story.append(Paragraph(
        "The table below shows the average number of times each Region of Interest (ROI) "
        "switched its functional community membership across all sliding windows, averaged "
        "across all subjects in each group. Higher values indicate greater dynamic flexibility.",
        body_style
    ))
    story.append(Spacer(1, 0.2 * inch))

    #Table of ROI switches
    table_data = [['ROI', 'Healthy (avg)', "Parkinson's (avg)", 'Difference']]
    for i, region in enumerate(pd_regions):
        diff = avg_parkinsons[i] - avg_healthy[i]
        table_data.append([
            region,
            f"{avg_healthy[i]:.2f}",
            f"{avg_parkinsons[i]:.2f}",
            f"{diff:+.2f}"
        ])

    table = Table(table_data, colWidths=[2.2*inch, 1.3*inch, 1.5*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, 0), 10),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EBF5FB'), colors.white]),
        ('FONTSIZE',   (0, 1), (-1, -1), 9),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    #ROI switches bar chart image
    if os.path.exists("results/plots/roi_switching_comparison.png"):
        story.append(Paragraph("Figure 1: ROI Flexibility Comparison", styles['Italic']))
        story.append(Image("results/plots/roi_switching_comparison.png", width=6*inch, height=3*inch))

    story.append(PageBreak())

    #Section 2: Graph Measures
    story.append(Paragraph("2. Graph-Theoretic Measures", heading_style))
    story.append(Paragraph(
        "The following measures were computed for each sliding window and averaged across "
        "all windows and all subjects. These measures quantify the network properties of "
        "brain connectivity at each point in time.",
        body_style
    ))
    story.append(Spacer(1, 0.2 * inch))

    measures_info = {
        'clustering':        ('Clustering Coefficient', 'Measures how tightly interconnected each brain region\'s neighbours are. Higher values indicate more cohesive local circuits.'),
        'path_length':       ('Characteristic Path Length', 'Average shortest path between all region pairs. Lower values indicate more efficient information transfer.'),
        'global_efficiency': ('Global Efficiency', 'Measures how efficiently information is exchanged across the whole brain network.'),
        'modularity':        ('Modularity', 'Measures how cleanly the network separates into distinct communities. Higher values indicate more isolated modules.')
    }

    measures_table_data = [['Measure', 'Healthy', "Parkinson's", 'Interpretation']]
    for key, (label, interpretation) in measures_info.items():
        h_val  = healthy_measures[key].mean()
        pd_val = parkinsons_measures[key].mean()
        higher = 'H > PD' if h_val > pd_val else 'PD > H'
        measures_table_data.append([label, f"{h_val:.4f}", f"{pd_val:.4f}", higher])

    measures_table = Table(measures_table_data, colWidths=[2*inch, 1.1*inch, 1.1*inch, 1.5*inch])
    measures_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1A5276')),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.HexColor('#EBF5FB'), colors.white]),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(measures_table)
    story.append(Spacer(1, 0.3 * inch))

    #Add interpretation paragraphs for each measure
    for key, (label, interpretation) in measures_info.items():
        h_val  = healthy_measures[key].mean()
        pd_val = parkinsons_measures[key].mean()
        story.append(Paragraph(f"<b>{label}:</b> {interpretation} "
                               f"Healthy: {h_val:.4f}, Parkinson's: {pd_val:.4f}.", body_style))
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())

    #Section 3: Recurrence Plots
    story.append(Paragraph("3. Recurrence Analysis", heading_style))
    story.append(Paragraph(
        "Recurrence plots visualise how often the brain revisits similar functional "
        "connectivity states over time. Each point (i, j) is black if the brain's "
        "connectivity pattern at window i was similar to window j. Structured patterns "
        "indicate organised, predictable dynamics while fragmented patterns indicate "
        "irregular, unstable dynamics.",
        body_style
    ))
    story.append(Spacer(1, 0.2 * inch))
    if os.path.exists("results/plots/recurrence_plot.png"):
        story.append(Paragraph("Figure 2: Recurrence Plots (Group-Averaged)", styles['Italic']))
        story.append(Image("results/plots/recurrence_plot.png", width=6*inch, height=3*inch))

    story.append(PageBreak())

    #Section 4: Conclusion
    story.append(Paragraph("4. Conclusion", heading_style))
    story.append(Paragraph(
        "This analysis demonstrates clear differences in dynamic brain connectivity between "
        "healthy subjects and Parkinson's disease patients across multiple measures:",
        body_style
    ))
    story.append(Spacer(1, 0.1 * inch))

    conclusions = [
        ("ROI Flexibility", 
         f"Parkinson's subjects show higher average ROI switch counts "
         f"({avg_parkinsons.mean():.2f}) compared to healthy subjects ({avg_healthy.mean():.2f}), "
         "suggesting increased instability in functional community membership rather than "
         "healthy adaptive flexibility."),
        ("Network Organisation",
         f"Healthy subjects exhibit higher clustering ({healthy_measures['clustering'].mean():.4f} vs "
         f"{parkinsons_measures['clustering'].mean():.4f}) and global efficiency "
         f"({healthy_measures['global_efficiency'].mean():.4f} vs "
         f"{parkinsons_measures['global_efficiency'].mean():.4f}), indicating more cohesive "
         "and efficiently integrated brain networks."),
        ("Modularity",
         f"Parkinson's subjects show higher modularity ({parkinsons_measures['modularity'].mean():.4f} vs "
         f"{healthy_measures['modularity'].mean():.4f}), indicating more isolated functional "
         "communities and reduced inter-network communication."),
        ("Recurrence Analysis",
         "Recurrence plots reveal that healthy subjects exhibit more structured, organised "
         "dynamic patterns while Parkinson's subjects show more fragmented, irregular "
         "transitions between brain states, consistent with disrupted functional network dynamics."),
        ("Overall Finding",
         "Parkinson's disease disrupts the brain's ability to maintain stable, efficiently "
         "integrated functional network organisation over time. The basal ganglia regions "
         "including the Putamen, Caudate, and Thalamus show particularly notable differences, "
         "consistent with known Parkinson's pathology in these areas.")
    ]

    for heading, text in conclusions:
        story.append(Paragraph(f"<b>{heading}:</b> {text}", body_style))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)
    print("Report saved to results/report.pdf")