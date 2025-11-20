"""
Create Figure 1: Temporal Tier Structure Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Tier 0 (Baseline) - Blue boxes
tier0_vars = ['Age', 'Gender', 'Risk\nScore', 'Baseline\nED Visits', 'Baseline\nIP Admits', 'Baseline\nCosts']
tier0_y = 0.2
for i, var in enumerate(tier0_vars):
    x = i * 2.2 + 0.5
    rect = patches.FancyBboxPatch((x, tier0_y), 1.8, 1.2,
                                   boxstyle="round,pad=0.15",
                                   edgecolor='#2E86AB', facecolor='#A9D6E5',
                                   linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x + 0.9, tier0_y + 0.6, var, ha='center', va='center',
            fontsize=10, fontweight='bold')

# Add tier label
ax.text(-0.5, tier0_y + 0.6, 'TIER 0\nBaseline\n(6 months\npre-activation)',
        ha='right', va='center', fontsize=11, fontweight='bold',
        color='#2E86AB')

# Tier 1 (Treatment) - Green boxes
tier1_vars = ['Therapy\nExposure', 'Pharmacy\nContacts', 'CHW\nContacts', 'Care Coord\nContacts']
tier1_y = 3.5
for i, var in enumerate(tier1_vars):
    x = i * 3.0 + 1.5
    rect = patches.FancyBboxPatch((x, tier1_y), 2.5, 1.2,
                                   boxstyle="round,pad=0.15",
                                   edgecolor='#2D6A4F', facecolor='#95D5B2',
                                   linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x + 1.25, tier1_y + 0.6, var, ha='center', va='center',
            fontsize=10, fontweight='bold')

# Add tier label
ax.text(-0.5, tier1_y + 0.6, 'TIER 1\nTreatment\n(30 days -\n6 months\npost-activation)',
        ha='right', va='center', fontsize=11, fontweight='bold',
        color='#2D6A4F')

# Tier 2 (Outcomes) - Orange boxes
tier2_vars = ['Followup\nED Visits', 'Followup\nIP Admits', 'Followup\nCosts']
tier2_y = 6.8
for i, var in enumerate(tier2_vars):
    x = i * 4.0 + 2.5
    rect = patches.FancyBboxPatch((x, tier2_y), 3.0, 1.2,
                                   boxstyle="round,pad=0.15",
                                   edgecolor='#D77A61', facecolor='#FFB4A2',
                                   linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x + 1.5, tier2_y + 0.6, var, ha='center', va='center',
            fontsize=10, fontweight='bold')

# Add tier label
ax.text(-0.5, tier2_y + 0.6, 'TIER 2\nOutcomes\n(6 months\npost-activation)',
        ha='right', va='center', fontsize=11, fontweight='bold',
        color='#D77A61')

# Add allowed arrows (temporal precedence)
# Tier 0 → Tier 1
arrow1 = patches.FancyArrowPatch((3, tier0_y + 1.2), (3, tier1_y - 0.1),
                                  arrowstyle='->', mutation_scale=30,
                                  linewidth=3, color='black', alpha=0.6)
ax.add_patch(arrow1)
ax.text(3.5, 2.3, 'Allowed\n(earlier → later)', fontsize=9, style='italic')

# Tier 1 → Tier 2
arrow2 = patches.FancyArrowPatch((6, tier1_y + 1.2), (6, tier2_y - 0.1),
                                  arrowstyle='->', mutation_scale=30,
                                  linewidth=3, color='black', alpha=0.6)
ax.add_patch(arrow2)

# Tier 0 → Tier 2 (baseline persistence)
arrow3 = patches.FancyArrowPatch((9, tier0_y + 1.2), (9, tier2_y - 0.1),
                                  arrowstyle='->', mutation_scale=30,
                                  linewidth=2.5, color='black', alpha=0.4,
                                  linestyle='dashed')
ax.add_patch(arrow3)
ax.text(9.8, 4.0, 'Allowed\n(persistence)', fontsize=9, style='italic')

# Add forbidden arrow (Tier 2 → Tier 0) with X
forbidden_x_start = 11.5
forbidden_y_start = tier2_y + 0.6
forbidden_y_end = tier0_y + 0.6
ax.plot([forbidden_x_start, forbidden_x_start], [forbidden_y_start, forbidden_y_end],
        'r--', linewidth=2.5, alpha=0.7)
# Big red X
ax.plot([forbidden_x_start - 0.4, forbidden_x_start + 0.4],
        [4.5 - 0.4, 4.5 + 0.4], 'r', linewidth=4)
ax.plot([forbidden_x_start - 0.4, forbidden_x_start + 0.4],
        [4.5 + 0.4, 4.5 - 0.4], 'r', linewidth=4)
ax.text(forbidden_x_start + 1.2, 4.5, 'Forbidden\n(later → earlier)',
        fontsize=9, style='italic', color='red')

ax.set_xlim(-1.5, 14)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')
ax.set_title('Temporal Tier Structure for Causal Discovery',
             fontsize=16, fontweight='bold', pad=20)

# Add caption box at bottom
caption = ("Variables organized into temporal tiers ensure causal ordering. Arrows indicate allowed causal relationships " +
           "(causes precede effects). Temporal precedence constraints forbid edges from later to earlier tiers, " +
           "eliminating reverse causation and reducing search space for causal discovery algorithms.")
ax.text(7, -0.3, caption, ha='center', va='top', fontsize=9,
        style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.5',
                                             facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/Users/sanjaybasu/waymark-local/results/causal_discovery_expanded/temporal_tiers_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 saved: temporal_tiers_diagram.png")
plt.close()
