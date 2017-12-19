from matplotlib.patches import Patch

def create_legend_labels(options):
    legend_labels = []
    for option in options:
        legend_labels.append(Patch(color=option['color'], label=option['label']))
    return legend_labels