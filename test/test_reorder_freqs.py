import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic.calculate.dispersion import reorder_freqs


class TestReorderFreqs(unittest.TestCase):

    def test_reorder_freqs_NaH(self):
        seedname = 'NaH-reorder-test'
        path = 'test/data'
        data = PhononData(seedname, path)
        data.convert_e_units('1/cm')
        reorder_freqs(data)
        expected_reordered_freqs = np.array(
            [[91.847109, 91.847109, 166.053018,
              564.508299, 564.508299, 884.068976],
             [154.825631, 132.031513, 206.21394,
              642.513551, 690.303338, 832.120011],
             [106.414367, 106.414367, 166.512415,
              621.498613, 621.498613, 861.71391],
             [-4.05580000e-02, -4.05580000e-02,
              1.23103200e+00, 5.30573108e+02,
              5.30573108e+02, 8.90673361e+02],
             [139.375186, 139.375186, 207.564309,
              686.675791, 686.675791, 833.291584],
             [123.623059, 152.926351, 196.644517,
              586.674239, 692.696132, 841.62725],
             [154.308477, 181.239973, 181.239973,
              688.50786, 761.918164, 761.918164],
             [124.976823, 124.976823, 238.903818,
              593.189877, 593.189877, 873.903056]])
        npt.assert_allclose(data.freqs.magnitude, expected_reordered_freqs)

    def test_reorder_freqs_LZO(self):
        seedname = 'La2Zr2O7'
        path = 'test/data'
        data = PhononData(seedname, path)
        data.convert_e_units('1/cm')
        reorder_freqs(data)

        expected_reordered_freqs = np.array(
            [[65.062447, 65.062447, 70.408176, 76.847761, 76.847761,
              85.664054, 109.121893, 109.121893, 117.920003, 119.363588,
              128.637195, 128.637195, 155.905812, 155.905812, 160.906969,
              170.885818, 172.820917, 174.026075, 178.344487, 183.364621,
              183.364621, 199.25343, 199.25343, 222.992334, 225.274444,
              231.641854, 253.012884, 265.452117, 270.044891, 272.376357,
              272.376357, 275.75891, 299.890562, 299.890562, 315.067652,
              315.067652, 319.909059, 338.929562, 338.929562, 339.067304,
              340.308461, 349.793091, 376.784786, 391.288446, 391.288446,
              396.109935, 408.179774, 408.179774, 410.991152, 421.254131,
              456.215732, 456.215732, 503.360953, 532.789756, 532.789756,
              545.400861, 548.704226, 552.622463, 552.622463, 557.488238,
              560.761581, 560.761581, 618.721858, 734.650232, 739.200593,
              739.200593],
             [62.001197, 62.001197, 67.432601, 70.911126, 70.911126,
              87.435181, 109.893289, 109.893289, 110.930712, 114.6143,
              129.226412, 129.226412, 150.593502, 150.593502, 148.065107,
              165.856823, 168.794942, 167.154743, 169.819174, 187.349434,
              187.349434, 202.003734, 202.003734, 221.329787, 231.797486,
              228.999412, 259.308314, 264.453017, 279.078288, 270.15176,
              270.15176, 278.861064, 300.349651, 300.349651, 311.929653,
              311.929653, 318.251662, 334.967743, 334.967743, 340.747776,
              338.357732, 356.048074, 372.658152, 395.526156, 395.526156,
              398.356528, 406.398552, 406.398552, 407.216469, 421.122741,
              460.527859, 460.527859, 486.346855, 533.694179, 533.694179,
              544.93361, 549.252501, 550.733812, 550.733812, 558.006939,
              559.641583, 559.641583, 591.170512, 739.589673, 738.563124,
              738.563124],
             [55.889266, 55.889266, 64.492348, 66.375741, 66.375741,
              88.940906, 109.388591, 109.388591, 100.956751, 109.379914,
              130.017598, 130.017598, 145.579207, 145.579207, 134.563651,
              161.166842, 164.427227, 159.401681, 161.563336, 190.735683,
              190.735683, 205.550607, 205.550607, 219.351563, 238.204625,
              226.878861, 265.686284, 263.148071, 287.722953, 267.983859,
              267.983859, 281.041577, 299.480498, 299.480498, 308.176127,
              308.176127, 318.101514, 332.930623, 332.930623, 344.002317,
              335.480119, 361.930637, 368.350971, 399.050499, 399.050499,
              399.241143, 404.639113, 404.639113, 400.809087, 420.335936,
              465.504468, 465.504468, 470.205579, 534.544778, 534.544778,
              544.501022, 549.755212, 548.80696, 548.80696, 556.193672,
              558.101279, 558.101279, 565.776342, 741.372005, 737.860626,
              737.860626],
             [46.935517, 46.935517, 61.690137, 63.177342, 63.177342,
              90.180632, 107.721223, 107.721223, 86.944159, 104.159787,
              130.879196, 130.879196, 141.295304, 141.295304, 122.536218,
              157.146893, 160.037586, 151.613374, 153.750028, 193.160653,
              193.160653, 209.882364, 209.882364, 215.936117, 244.178665,
              225.432553, 272.052764, 261.655838, 295.533954, 265.906764,
              265.906764, 282.006965, 295.142911, 295.142911, 307.16826,
              307.16826, 319.295877, 332.071847, 332.071847, 348.814514,
              332.065989, 367.152249, 364.288189, 400.773283, 400.773283,
              399.790407, 404.068253, 404.068253, 387.165977, 418.829125,
              470.716023, 470.716023, 460.278318, 535.223077, 535.223077,
              544.111882, 550.193478, 547.016352, 547.016352, 552.362689,
              556.261571, 556.261571, 543.678775, 740.965394, 737.162508,
              737.162508],
             [36.367201, 36.367201, 59.168434, 60.36167, 60.36167,
              91.154677, 105.37576, 105.37576, 68.755044, 99.446481,
              131.658334, 131.658334, 138.017877, 138.017877, 113.14576,
              153.975056, 156.016054, 144.576942, 146.47047, 194.581347,
              194.581347, 214.716315, 214.716315, 210.473211, 249.235088,
              224.769091, 278.102009, 260.171794, 302.032435, 263.72796,
              263.72796, 282.018114, 289.408098, 289.408098, 308.097577,
              308.097577, 321.241146, 331.659808, 331.659808, 353.492915,
              328.675778, 371.468173, 362.406897, 399.901709, 399.901709,
              399.179346, 405.625572, 405.625572, 368.236337, 416.52493,
              475.665346, 475.665346, 458.944007, 535.667484, 535.667484,
              543.78033, 550.551048, 545.494533, 545.494533, 547.179463,
              554.338811, 554.338811, 524.846465, 739.380608, 736.536495,
              736.536495],
             [24.785718, 24.785718, 57.117299, 57.830885, 57.830885,
              91.859898, 103.047316, 103.047316, 47.456331, 95.691927,
              132.248074, 132.248074, 135.79383, 135.79383, 106.389552,
              151.718169, 152.772977, 138.984268, 139.88209, 195.244028,
              195.244028, 219.466615, 219.466615, 203.707835, 252.993107,
              224.615517, 283.248783, 258.912028, 306.841458, 261.246129,
              261.246129, 281.584343, 284.696598, 284.696598, 309.37963,
              309.37963, 323.205545, 331.373295, 331.373295, 353.088149,
              326.000428, 374.686778, 367.331006, 398.738183, 398.738183,
              398.433921, 407.157219, 407.157219, 349.637392, 413.438689,
              479.806857, 479.806857, 463.608166, 535.889622, 535.889622,
              543.524255, 550.815232, 544.325882, 544.325882, 541.757933,
              552.630089, 552.630089, 508.677347, 737.533584, 736.042236,
              736.042236],
             [12.555025, 12.555025, 55.757043, 55.972359, 55.972359,
              92.288749, 101.380298, 101.380298, 24.214202, 93.270077,
              132.593517, 132.593517, 134.540163, 134.540163, 102.211134,
              150.378051, 150.665566, 135.38769, 134.747421, 195.473725,
              195.473725, 223.210107, 223.210107, 197.85154, 255.276828,
              224.659659, 286.758828, 258.068085, 309.776254, 258.846604,
              258.846604, 281.147316, 281.874849, 281.874849, 310.385381,
              310.385381, 324.609898, 331.158402, 331.158402, 351.072968,
              324.818103, 376.67194, 374.186388, 397.950964, 397.950964,
              397.878833, 408.114477, 408.114477, 336.37863, 410.112489,
              482.591747, 482.591747, 471.735469, 535.964134, 535.964134,
              543.361599, 550.977172, 543.571634, 543.571634, 537.566668,
              551.451065, 551.451065, 494.626062, 736.133131, 735.72609,
              735.72609],
             [-0.019621, -0.019621, 55.277927, 55.277927, 55.277927,
              92.432911, 100.780857, 100.780857, -0.019621, 92.432911,
              132.696363, 132.696363, 134.147102, 134.147102, 100.780857,
              149.934817, 149.934817, 134.147102, 132.696363, 195.519690,
              195.519690, 224.698049, 224.698049, 195.519690, 256.039866,
              224.698049, 288.011070, 257.771213, 310.763767, 257.771213,
              257.771213, 280.972846, 280.972846, 280.972846, 310.763767,
              310.763767, 325.114540, 331.073494, 331.073494, 350.234619,
              325.114540, 377.342620, 377.342620, 397.677533, 397.677533,
              397.677533, 408.435923, 408.435923, 331.073494, 408.435923,
              483.578389, 483.578389, 480.948578, 535.976810, 535.976810,
              543.305729, 551.031712, 543.305729, 543.305729, 535.976810,
              551.031712, 551.031712, 483.578389, 735.617369,
              735.617369, 735.617369],
             [12.555025, 12.555025, 55.757043, 55.972359, 55.972359,
              92.288749, 101.380298, 101.380298, 24.214202, 93.270077,
              132.593517, 132.593517, 134.540163, 134.540163, 102.211134,
              150.378051, 150.665566, 135.38769, 134.747421, 195.473725,
              195.473725, 223.210107, 223.210107, 197.85154, 255.276828,
              224.659659, 286.758828, 258.068085, 309.776254, 258.846604,
              258.846604, 281.147316, 281.874849, 281.874849, 310.385381,
              310.385381, 324.609898, 331.158402, 331.158402, 351.072968,
              324.818103, 376.67194, 374.186388, 397.950964, 397.950964,
              397.878833, 408.114477, 408.114477, 336.37863, 410.112489,
              482.591747, 482.591747, 471.735469, 535.964134, 535.964134,
              543.361599, 550.977172, 543.571634, 543.571634, 537.566668,
              551.451065, 551.451065, 494.626062, 736.133131, 735.72609,
              735.72609],
             [24.785718, 24.785718, 57.117299, 57.830885, 57.830885,
              91.859898, 103.047316, 103.047316, 47.456331, 95.691927,
              132.248074, 132.248074, 135.79383, 135.79383, 106.389552,
              151.718169, 152.772977, 138.984268, 139.88209, 195.244028,
              195.244028, 219.466615, 219.466615, 203.707835, 252.993107,
              224.615517, 283.248783, 258.912028, 306.841458, 261.246129,
              261.246129, 281.584343, 284.696598, 284.696598, 309.37963,
              309.37963, 323.205545, 331.373295, 331.373295, 353.088149,
              326.000428, 374.686778, 367.331006, 398.738183, 398.738183,
              398.433921, 407.157219, 407.157219, 349.637392, 413.438689,
              479.806857, 479.806857, 463.608166, 535.889622, 535.889622,
              543.524255, 550.815232, 544.325882, 544.325882, 541.757933,
              552.630089, 552.630089, 508.677347, 737.533584, 736.042236,
              736.042236]])
        npt.assert_allclose(data.freqs.magnitude, expected_reordered_freqs)

    def test_multiple_reorder_freqs_has_no_effect_LZO(self):
        # Test that when reorder_freqs is called more than once on the same
        # object, the second time it has no effect, as the frequencies have
        # already been reordered. This ensures the eigenvectors have also
        # been reordered correctly.
        seedname = 'La2Zr2O7'
        path = 'test/data'
        data = PhononData(seedname, path)
        reorder_freqs(data)
        reordered_freqs_1 = data.freqs.magnitude
        reorder_freqs(data)
        reordered_freqs_2 = data.freqs.magnitude

        npt.assert_array_equal(reordered_freqs_1, reordered_freqs_2)
