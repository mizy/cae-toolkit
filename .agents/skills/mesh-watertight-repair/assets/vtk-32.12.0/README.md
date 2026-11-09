# Offline report renderer

Unmodified UMD bundle from the public `vtk.js` npm package, version 32.12.0:
https://registry.npmjs.org/vtk.js/-/vtk.js-32.12.0.tgz

`vtk.js` SHA-256: `8e533c77cece64afa371b7700a797e24a6d7e05a8f58b2c2b5d8b415d67ea09c`.
The package tarball was verified against the npm registry's SHA-512 integrity.

`html_mesh_preview.vtk_js_source()` embeds this bundle in each HTML report so
opening a report requires no CDN or report-time download. `VTK_JS_PATH` can
override it with an explicit local bundle. Report surface rendering does not
use the optional vtk.js workers or source map, which are not packaged here.

See LICENSE, Copyright.txt, and vtk.js.LICENSE.txt for upstream notices.
