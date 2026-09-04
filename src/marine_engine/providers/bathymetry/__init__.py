"""Bathymetry source discovery, spatial verification, and raw acquisition.

Each approved source family (`ukho`, `bgs`, `emodnet`) is a concrete module
that returns raw `inventory.SurveyRecord` candidates; `inventory.py` is
where those candidates get normalized, spatially verified against the
canonical PL854 pipeline/AOI/chainage, and ranked. `acquisition.py` handles
downloading and manifesting whichever candidates are automatically
accessible.
"""
