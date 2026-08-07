from wp_gen import *

def gsd_to_lane_spacing_m(
    gsd_cm_per_px: float,
    sidelap_percent: float,
    image_width_px: int = 6000,
) -> float:
    """
    Convert GSD to lane spacing.

    lane_spacing_m = GSD(m/px) * image_width(px) * (1 - sidelap)
    """
    return (gsd_cm_per_px / 100.0) * image_width_px * (1.0 - sidelap_percent / 100.0)


# Choose ONE:
# 1) GSD-based list
gsd_list = [10, 7.5, 5, 2.5, 1]   # cm/px

# 2) Or direct lane spacing list
# lane_list = [168.0, 111.0, 86.0]

sidelap_percent = 60.0
image_width_px = 6000

# Convert GSD list -> lane spacing list
lane_distances_m = [
    gsd_to_lane_spacing_m(gsd, sidelap_percent, image_width_px)
    for gsd in gsd_list
]
print(lane_distances_m)

mission = generate_qgc_wpl_110_mission(
    lane_length_m=500.0,
    lane_orientation_deg=270.0,   # first lane goes right -> left
    lane_distances_m=lane_distances_m,
)

write_qgc_wpl_110("generated.waypoints", mission)
