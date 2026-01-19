# Quality gates (repo-local)

QG1. No normative content required
- Implementation: content is technical and algorithmic; no ethical/political content.

QG2. Assumption register with falsification probes
- Implementation: `docs/ASSUMPTIONS.md` plus PROVENANCE Assumption Register.

QG3. Full requirement coverage
- Implementation: project includes complete pipeline wiring with build system, CLI, serialization, SfM, BA, PMVS/Bundler/PLY exports.

QG4. Units/calculations consistent
- Implementation: pixel units for image-space errors and coordinates; SI units implicit for world space (scale is gauge-dependent).

QG5. Edge cases addressed
- Implementation: explicit failure paths for missing images, missing verified pairs, insufficient correspondences.

QG6. Verified provenance
- Implementation: PROVENANCE section lists primary algorithm sources and dependency licenses.

QG7. Bounded scope expansion complete
- Implementation: PROVENANCE section lists high/medium/low impact extensions and risks.