"""Native reference-grid and spectral-product export regressions."""
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from seleno.tool.scene import Scene, NativeBand, load
from seleno.tool.export import reference_to_source, sample_band, write_registered_native
from seleno.tool.coordinates import grid_to_reference, project
from seleno.tool import warp_model


class NativeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.crs = CRS.from_proj4("+proj=stere +lat_0=-90 +R=1737400 +units=m")

    def scene(self, data, **kwargs):
        return Scene(path="array", array=np.asarray(data), valid=None,
                     reader="test", nodata=None, **kwargs)

    def test_all_invalid_export_tile_has_no_nanmedian_warning(self):
        scene = self.scene(np.ones((8, 8), np.float32))
        with patch("seleno.tool.export.reference_to_source",
                   side_effect=lambda s, r, f, p, **kw: np.full_like(p, np.nan)):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                info = write_registered_native(self.root, scene, scene, {},
                                                {"matrix": np.eye(3).tolist()})
        self.assertEqual(info["supersampling"], [1])
        with rasterio.open(self.root / "registered.tif") as ds:
            self.assertTrue(np.isnan(ds.read(1)).all())

    def test_decimated_fit_exports_original_resolution_and_detail(self):
        # Every native pixel differs, so enlarging a 4x working image fails.
        data = np.arange(79 * 83, dtype=np.float32).reshape(79, 83)
        transform = Affine(1, .2, 100, .1, -1, 500)
        scene = self.scene(data, transform=transform, crs=self.crs)
        frame = {"reference_decimation": 4, "reference_origin": [7, 9],
                 "reference_window": [7, 9, 73, 80],
                 "reference_sample_offset": [1.5, 1.5]}
        info = write_registered_native(self.root, scene, scene, frame,
                                       {"matrix": np.eye(3).tolist()}, tile_size=17)
        self.assertEqual(info["shape"], [66, 71])
        self.assertEqual(info["reference_decimation"], 1)
        with rasterio.open(self.root / "registered.tif") as ds:
            np.testing.assert_array_equal(ds.read(1), data[7:73, 9:80])
            self.assertEqual(ds.transform, transform * Affine.translation(9, 7))

    def test_complete_segment_field_samples_original_cube(self):
        yy, xx = np.mgrid[:90, :70]
        bands = tuple(NativeBand((xx + 3 * yy + i * 1000).astype(np.float32))
                      for i in range(3))
        source = self.scene(bands[0].array, native_bands=bands)
        frame = {"reference_decimation": 3, "reference_sample_offset": [1., 1.]}
        first, second = np.eye(3), np.eye(3)
        first[:2, 2], second[:2, 2] = [.15, -.2], [.3, .4]
        model = {"matrix": np.eye(3).tolist(), "segments": [
            {"axis": "row", "from": 0, "to": 15, "matrix": first.tolist()},
            {"axis": "row", "from": 15, "to": 30, "matrix": second.tolist()}]}
        write_registered_native(self.root, source, source, frame, model, tile_size=16)
        points = np.array([[20., 20.], [30., 44.], [40., 75.]])
        c = grid_to_reference(frame)
        native = project(c, warp_model.inverse_points(model, project(np.linalg.inv(c), points)))
        with rasterio.open(self.root / "registered.tif") as ds:
            self.assertEqual(ds.count, 3)
            for i in range(3):
                expected = native[:, 0] + 3 * native[:, 1] + i * 1000
                actual = ds.read(i + 1)[points[:, 1].astype(int), points[:, 0].astype(int)]
                np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_geotiff_retains_bands_masks_wavelengths_and_radiometry(self):
        data = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
        path = self.root / "cube.tif"
        with rasterio.open(path, "w", driver="GTiff", width=40, height=32,
                           count=3, dtype="uint16", crs=self.crs,
                           transform=Affine(2, 0, 100, 0, -2, 100)) as ds:
            for band in range(1, 4):
                ds.write(data * band, band)
                ds.set_band_description(band, "spectral %d" % band)
                ds.set_band_unit(band, "radiance")
                ds.update_tags(band, wavelength_um=str(.8 + band * .2))
            ds.scales = (.1, .2, .3)
            ds.offsets = (1., 2., 3.)
            mask = np.full(data.shape, 255, np.uint8)
            mask[10:15, 10:15] = 0
            ds.write_mask(mask)
        scene = load(str(path))
        self.assertEqual(scene.summary()["bands"], 3)
        write_registered_native(self.root, scene, scene, {}, {"matrix": np.eye(3).tolist()})
        with rasterio.open(self.root / "registered.tif") as ds:
            self.assertEqual(ds.scales, (.1, .2, .3))
            self.assertEqual(ds.offsets, (1., 2., 3.))
            self.assertEqual(ds.descriptions, ("spectral 1", "spectral 2", "spectral 3"))
            self.assertEqual(ds.units, ("radiance",) * 3)
            for band in range(1, 4):
                result = ds.read(band)
                np.testing.assert_array_equal(result[:8], data[:8] * band)
                self.assertTrue(np.isnan(result[10:15, 10:15]).all())
                self.assertEqual(ds.tags(band)["wavelength_um"], str(.8 + band * .2))
            self.assertEqual(ds.read(1)[0, 0], 0)  # Valid black is not nodata.

    def test_pds4_interleaves_preserve_every_band_and_header_offset(self):
        cube = (np.arange(3 * 32 * 40).reshape(3, 32, 40) + 1).astype("<u2")
        for order, axes, shape in (("bsq", ["Band", "Line", "Sample"], cube),
                                   ("bil", ["Line", "Band", "Sample"], cube.transpose(1, 0, 2)),
                                   ("bip", ["Line", "Sample", "Band"], cube.transpose(1, 2, 0))):
            with self.subTest(order=order):
                raw, label = self.root / (order + ".qub"), self.root / (order + ".xml")
                raw.write_bytes(b"header bytes" + shape.tobytes())
                dimensions = dict(Band=3, Line=32, Sample=40)
                axes_xml = "".join("<Axis_Array><axis_name>%s</axis_name><elements>%d</elements></Axis_Array>"
                                   % (a, dimensions[a]) for a in axes)
                label.write_text(
                    "<Product_Observational><logical_identifier>urn:isro:ch2_iir:test</logical_identifier>"
                    "<file_name>%s</file_name><Array_3D_Spectrum><offset>12</offset>"
                    "<data_type>UnsignedLSB2</data_type><axis_index_order>Last Index Fastest</axis_index_order>%s"
                    "</Array_3D_Spectrum><center_value unit=\"micrometer\">0.9</center_value>"
                    "<center_value unit=\"micrometer\">1.2</center_value>"
                    "<center_value unit=\"micrometer\">1.5</center_value></Product_Observational>"
                    % (raw.name, axes_xml))
                scene = load(str(label))
                self.assertEqual(len(scene.bands), 3)
                for i, band in enumerate(scene.bands):
                    np.testing.assert_array_equal(band.array, cube[i])
                    self.assertEqual(float(band.tags["wavelength_um"]), [.9, 1.2, 1.5][i])
                write_registered_native(self.root, scene, scene, {}, {"matrix": np.eye(3).tolist()})
                with rasterio.open(self.root / "registered.tif") as ds:
                    np.testing.assert_array_equal(ds.read(), cube)

    def test_coordinate_routes_use_centres_and_do_not_clamp_to_working_grid(self):
        a = np.ones((32, 40), np.float32)
        points = np.array([[0., 0.], [20.25, 17.5], [39., 31.]])
        src, ref = self.scene(a, gsd_m=.25), self.scene(a, gsd_m=1.)
        np.testing.assert_allclose(reference_to_source(src, ref, {}, points), (points + .5) * 4 - .5)
        placement = [[0., -2., 100.], [2., 0., 10.], [0., 0., 1.]]
        np.testing.assert_allclose(reference_to_source(src, ref, {"placement": placement}, points),
                                   project(placement, points))
        src = self.scene(a, crs=self.crs, transform=Affine(.25, 0, 100, 0, -.25, 200))
        ref = self.scene(a, crs=self.crs, transform=Affine(1, 0, 100, 0, -1, 200))
        np.testing.assert_allclose(reference_to_source(src, ref, {}, points), (points + .5) * 4 - .5)
        src.lonlat = object()
        calls = []

        def interpolators(grid, crs, cache):
            calls.append(crs)
            return lambda x, y: 4 * (x - 100) - .5, lambda x, y: -4 * (y - 200) - .5, lambda x, y: (x, y)

        np.testing.assert_allclose(reference_to_source(src, ref, {}, points, lattice_interpolators=interpolators),
                                   (points + .5) * 4 - .5)
        self.assertEqual(calls, [self.crs])

    def test_bilinear_nodata_and_subpixel_interpolation(self):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        band = NativeBand(data, nodata=44)
        actual = sample_band(band, [[2.25, 3.5], [3, 4], [4, 4], [3.5, 4], [-1, 0], [9.25, 0]])
        np.testing.assert_allclose(actual[:2], [37.25, 43.])
        self.assertTrue(np.isnan(actual[2:5]).all())
        self.assertEqual(actual[5], 9.)

    def test_streaming_never_materialises_source_or_output_raster(self):
        class GuardedArray:
            shape = (310, 270)
            dtype = np.dtype("float32")

            def __array__(self, *args, **kwargs):
                raise AssertionError("whole source array was requested")

            def __getitem__(self, key):
                rows, cols = key
                self_test.assertLessEqual(np.size(rows), 23 ** 2)
                return np.asarray(rows * 270 + cols, np.float32)

        self_test = self
        source = self.scene(np.ones(GuardedArray.shape, np.float32), native_bands=(NativeBand(GuardedArray()),))
        write_registered_native(self.root, source, source, {"reference_decimation": 4},
                                {"matrix": np.eye(3).tolist()}, tile_size=23)
        with rasterio.open(self.root / "registered.tif") as ds:
            self.assertEqual(ds.shape, GuardedArray.shape)
            self.assertEqual(ds.read(1)[-1, -1], 310 * 270 - 1)


if __name__ == "__main__":
    unittest.main()
