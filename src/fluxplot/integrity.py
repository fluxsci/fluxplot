"""Referential checks against the SVG actually drawn, before any files are replaced."""
from __future__ import annotations


def validate_references(manifest, present):
    refs = set()
    for axis in manifest['axes']:
        if axis.get('svgId'): refs.add(axis['svgId'])
    for series in manifest['series']:
        for value in series['svg'].values():
            refs.update(value if isinstance(value, list) else [value])
        for part in series.get('components', []):
            refs.add(part['svgId']); refs.update(part.get('members', []))
        for point in series.get('points', []): refs.add(point['svgId'])
    for guide in manifest.get('guides', []):
        if guide.get('svgId'): refs.add(guide['svgId'])
        if guide.get('mappable'): refs.add(guide['mappable'])
        for part in guide.get('parts', []): refs.add(part['svgId'])
    for overlay in manifest.get('overlays', []): refs.add(overlay['svgId'])
    def walk(node):
        if node.get('ref'): refs.add(node['ref'])
        refs.update(node.get('members', []))
        for child in node.get('children', []): walk(child)
    walk(manifest['parts'])
    missing = refs - present
    if missing:
        raise ValueError('manifest references missing SVG parts: ' + ', '.join(sorted(missing)[:10]))
