"""
SLD99 Coordinate Transformation Module

Performs geodetic datum conversions and map projections:
WGS84 Geographic (Lat, Lon, Height) -> SLD99 Transverse Mercator Grid (Easting, Northing).
Conforms to EPSG:5235.
"""

import math

class SLD99Transformer:
    """
    WGS84 (φ, λ) -> SLD99 Transverse Mercator (Easting, Northing)
    Conforms to EPSG:5235 parameters.
    """
    _WGS84_a = 6_378_137.0
    _WGS84_f = 1.0 / 298.257_223_563
    _EVE_a   = 6_377_276.345
    _EVE_f   = 1.0 / 300.8017

    # Helmert 3-parameter translation shifts: WGS84 -> Everest 1830 (1937 Adjustment)
    _dX = +100.7
    _dY = -784.8
    _dZ = -86.2

    # Natural Origin parameters
    _phi0 = math.radians(7 + 1.698 / 3600)         # 7°00'01.698" N
    _lam0 = math.radians(80.7717130833333)         # 80°46'18.167" E
    _k0   = 0.9999238418                           # Central scale factor
    _FE   = 500_000.0                              # False Easting
    _FN   = 500_000.0                              # False Northing

    @classmethod
    def _ellipsoid_params(cls, a, f):
        b = a * (1.0 - f)
        e2 = 2 * f - f * f
        ep2 = e2 / (1.0 - e2)
        return b, e2, ep2

    @classmethod
    def _geo_to_xyz(cls, phi, lam, h, a, e2):
        N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
        X = (N + h) * math.cos(phi) * math.cos(lam)
        Y = (N + h) * math.cos(phi) * math.sin(lam)
        Z = (N * (1.0 - e2) + h) * math.sin(phi)
        return X, Y, Z

    @classmethod
    def _xyz_to_geo(cls, X, Y, Z, a, e2):
        b = a * math.sqrt(1.0 - e2)
        ep2 = e2 / (1.0 - e2)
        p = math.sqrt(X * X + Y * Y)
        lam = math.atan2(Y, X)
        theta = math.atan2(Z * a, p * b)
        phi = math.atan2(Z + ep2 * b * math.sin(theta) ** 3,
                         p - e2 * a * math.cos(theta) ** 3)
        for _ in range(10):
            N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
            phi_ = math.atan2(Z + e2 * N * math.sin(phi), p)
            if abs(phi_ - phi) < 1e-12:
                phi = phi_
                break
            phi = phi_
        N = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
        h = p / math.cos(phi) - N if abs(math.cos(phi)) > 1e-9 else \
            Z / math.sin(phi) - N * (1.0 - e2)
        return phi, lam, h

    @classmethod
    def _helmert_shift(cls, X, Y, Z):
        return X + cls._dX, Y + cls._dY, Z + cls._dZ

    @classmethod
    def _tm_forward(cls, phi, lam, a, e2):
        k0 = cls._k0
        phi0 = cls._phi0
        lam0 = cls._lam0
        FE = cls._FE
        FN = cls._FN
        n = (a - a * math.sqrt(1.0 - e2)) / (a + a * math.sqrt(1.0 - e2))

        def _M(phi_r):
            A0_ = 1.0 + n ** 2 / 4 + n ** 4 / 64
            A2_ = 3 / 2 * (n - n ** 3 / 8)
            A4_ = 15 / 16 * (n ** 2 - n ** 4 / 4)
            A6_ = 35 / 48 * n ** 3
            A8_ = 315 / 512 * n ** 4
            return a / (1 + n) * (A0_ * phi_r
                                  - A2_ * math.sin(2 * phi_r)
                                  + A4_ * math.sin(4 * phi_r)
                                  - A6_ * math.sin(6 * phi_r)
                                  + A8_ * math.sin(8 * phi_r))

        M0 = _M(phi0)
        M = _M(phi)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        tan_phi = math.tan(phi)
        N_ = a / math.sqrt(1.0 - e2 * sin_phi ** 2)
        T = tan_phi ** 2
        C = (e2 / (1.0 - e2)) * cos_phi ** 2
        A_ = cos_phi * (lam - lam0)
        ep2 = e2 / (1.0 - e2)

        # Standard TM Northing Projection formula (EPSG Method 9807)
        northing = k0 * (
            M - M0
            + N_ * tan_phi * A_ ** 2 / 2
            + N_ * tan_phi * (5 - T + 9 * C + 4 * C ** 2) * A_ ** 4 / 24
            + N_ * tan_phi * (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A_ ** 6 / 720
        ) + FN

        # Standard TM Easting Projection formula
        easting = k0 * N_ * (
            A_
            + (1 - T + C) * A_ ** 3 / 6
            + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A_ ** 5 / 120
        ) + FE

        return easting, northing

    @classmethod
    def transform(cls, lat_wgs84_deg, lon_wgs84_deg, h_wgs84=0.0):
        """Transform geodetic WGS84 coordinates to SLD99 grid projected coordinates."""
        phi_w = math.radians(lat_wgs84_deg)
        lam_w = math.radians(lon_wgs84_deg)
        b_w, e2_w, _ = cls._ellipsoid_params(cls._WGS84_a, cls._WGS84_f)
        b_e, e2_e, _ = cls._ellipsoid_params(cls._EVE_a, cls._EVE_f)
        Xw, Yw, Zw = cls._geo_to_xyz(phi_w, lam_w, h_wgs84, cls._WGS84_a, e2_w)
        Xe, Ye, Ze = cls._helmert_shift(Xw, Yw, Zw)
        phi_e, lam_e, h_e = cls._xyz_to_geo(Xe, Ye, Ze, cls._EVE_a, e2_e)
        E, N = cls._tm_forward(phi_e, lam_e, cls._EVE_a, e2_e)
        return {
            "SLD99_E": E,
            "SLD99_N": N,
            "eve_lat_deg": math.degrees(phi_e),
            "eve_lon_deg": math.degrees(lam_e),
            "eve_h_m": h_e,
            "wgs84_X_ecef": Xw,
            "wgs84_Y_ecef": Yw,
            "wgs84_Z_ecef": Zw,
        }
