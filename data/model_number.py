import pandas as pd

INPUT_FILE = "master_msrp.csv"
OUTPUT_FILE = "master_msrp_with_model_numbers.csv"

# ------------------------------------------------------------
# iPhone model number mapping
# ------------------------------------------------------------

iphone_model_numbers = {

    "iPhone": "A1203",

    "iPhone 5s":
        "A1453, A1457, A1518, A1528, A1530, A1533",

    "iPhone 6":
        "A1549, A1586, A1589",

    "iPhone 6 Plus":
        "A1522, A1524, A1593",

    "iPhone 6s":
        "A1633, A1688, A1700",

    "iPhone 6s Plus":
        "A1634, A1687, A1699",

    "iPhone SE (1st generation)":
        "A1662, A1723, A1724",

    "iPhone 7":
        "A1660, A1778, A1779",

    "iPhone 7 Plus":
        "A1661, A1784, A1785",

    "iPhone 8":
        "A1863, A1905, A1906",

    "iPhone 8 Plus":
        "A1864, A1897, A1898",

    "iPhone X":
        "A1865, A1901, A1902",

    "iPhone XR":
        "A1984, A2105, A2106, A2108",

    "iPhone XS":
        "A1920, A2097, A2098, A2099, A2100",

    "iPhone XS Max":
        "A1921, A2101, A2102, A2104",

    "iPhone 11":
        "A2111, A2221, A2223",

    "iPhone 11 Pro":
        "A2160, A2215, A2217",

    "iPhone 11 Pro Max":
        "A2161, A2218, A2220",

    "iPhone SE (2nd generation)":
        "A2275, A2298, A2296",

    "iPhone 12 mini":
        "A2176, A2398, A2399, A2400",

    "iPhone 12":
        "A2172, A2402, A2403, A2404",

    "iPhone 12 Pro":
        "A2341, A2406, A2407, A2408",

    "iPhone 12 Pro Max":
        "A2342, A2410, A2411, A2412",

    "iPhone 13 mini":
        "A2481, A2626, A2628, A2629, A2630",

    "iPhone 13":
        "A2482, A2631, A2633, A2634, A2635",

    "iPhone 13 Pro":
        "A2483, A2636, A2638, A2639, A2640",

    "iPhone 13 Pro Max":
        "A2484, A2641, A2643, A2644, A2645",

    "iPhone SE (3rd generation)":
        "A2595, A2782, A2783, A2784, A2785",

    "iPhone 14":
        "A2649, A2881, A2882, A2883, A2884",

    "iPhone 14 Plus":
        "A2632, A2885, A2886, A2887, A2888",

    "iPhone 14 Pro":
        "A2650, A2890, A2891, A2892, A2889",

    "iPhone 14 Pro Max":
        "A2651, A2893, A2894, A2895, A2896",

    "iPhone 15":
        "A2846, A3089, A3090, A3092",

    "iPhone 15 Plus":
        "A2847, A3093, A3094, A3096",

    "iPhone 15 Pro":
        "A2848, A3101, A3102, A3104",

    "iPhone 15 Pro Max":
        "A2849, A3105, A3106, A3108",

    "iPhone 16":
        "A3081, A3286, A3287, A3288",

    "iPhone 16 Plus":
        "A3082, A3289, A3290, A3291",

    "iPhone 16 Pro":
        "A3083, A3292, A3293, A3294",

    "iPhone 16 Pro Max":
        "A3084, A3295, A3296, A3297",

    "iPhone 16e":
        "A3212, A3408, A3409, A3410",

    "iPhone 17":
        "A3258, A3519, A3520, A3521",

    "iPhone 17 Pro":
        "A3256, A3522, A3523, A3524",

    "iPhone 17 Pro Max":
        "A3257, A3525, A3526, A3527",

    "iPhone Air":
        "A3260, A3516, A3517, A3518",

    "iPhone 17e":
        "A3575, A3634, A3635",
}

iphone_model_numbers.update({

    # Naming variants in the existing dataset
    "iPhone 12 Mini":
        iphone_model_numbers["iPhone 12 mini"],

    "iPhone 13 Mini":
        iphone_model_numbers["iPhone 13 mini"],

    "iPhone SE (2016)":
        iphone_model_numbers["iPhone SE (1st generation)"],

    "iPhone SE (2020)":
        iphone_model_numbers["iPhone SE (2nd generation)"],

    "iPhone SE (2022)":
        iphone_model_numbers["iPhone SE (3rd generation)"],


    # US / eSIM variants
    #
    # These are intentionally separate dataset names because
    # your dataset explicitly distinguishes them.

    "iPhone 14 (eSIM)":
        "A2649",

    "iPhone 14 Plus (eSIM)":
        "A2632",

    "iPhone 14 Pro (eSIM)":
        "A2650",

    "iPhone 14 Pro Max (eSIM)":
        "A2651",

    "iPhone 15 (eSIM)":
        "A2846",

    "iPhone 15 Plus (eSIM)":
        "A2847",

    "iPhone 15 Pro (eSIM)":
        "A2848",

    "iPhone 15 Pro Max (eSIM)":
        "A2849",

    "iPhone 16 (eSIM)":
        "A3081",

    "iPhone 16 Plus (eSIM)":
        "A3082",

    "iPhone 16 Pro (eSIM)":
        "A3083",

    "iPhone 16 Pro Max (eSIM)":
        "A3084",

    "iPhone 16e (eSIM)":
        "A3212",
})

# ------------------------------------------------------------
# iPad model number mapping
# ------------------------------------------------------------

ipad_model_numbers = {

    # --------------------------------------------------------
    # Standard iPad
    # --------------------------------------------------------

    "iPad 4":
        "A1458, A1459, A1460",

    "iPad 5":
        "A1822, A1823",

    "iPad 6":
        "A1893, A1954",

    "iPad 7":
        "A2197, A2198, A2200",

    "iPad 8":
        "A2270, A2428, A2429, A2430",

    "iPad 9":
        "A2602, A2603, A2604, A2605",

    "iPad 10":
        "A2696, A2757, A2777, A3162",

    "iPad 11":
        "A3354, A3355, A3356",


    # --------------------------------------------------------
    # iPad Air
    # --------------------------------------------------------

    "iPad Air 1 9.7-inch A7 (2013)":
        "A1474, A1475, A1476",

    "iPad Air 2 9.7-inch A8X (2014)":
        "A1566, A1567",

    "iPad Air 3 10.5-inch A12 (2019)":
        "A2152, A2123, A2153, A2154",

    "iPad Air 4 10.9-inch A14 (2020)":
        "A2316, A2324, A2325, A2072",

    "iPad Air 5 10.9-inch M1 (2022)":
        "A2588, A2589, A2591",

    "iPad Air 6 11-inch M2 (2024)":
        "A2902, A2903, A2904",

    "iPad Air 6 13-inch M2 (2024)":
        "A2898, A2899, A2900",

    "iPad Air 7 11-inch M3 (2025)":
        "A3266, A3267, A3270",

    "iPad Air 7 13-inch M3 (2025)":
        "A3268, A3269, A3271",


    # --------------------------------------------------------
    # iPad mini
    # --------------------------------------------------------

    "iPad Mini 1":
        "A1432, A1454, A1455",

    "iPad Mini 2":
        "A1489, A1490, A1491",

    "iPad Mini 3":
        "A1599, A1600",

    "iPad Mini 4":
        "A1538, A1550",

    "iPad Mini 5":
        "A2133, A2124, A2125, A2126",

    "iPad Mini 6":
        "A2567, A2568, A2569",

    # Your dataset calls the A17 Pro generation "Mini 7".
    "iPad Mini 7":
        "A2993, A2995, A2996",


    # --------------------------------------------------------
    # iPad Pro
    # --------------------------------------------------------

    "iPad Pro 9.7-inch A9X (2016)":
        "A1673, A1674, A1675",

    "iPad Pro 10.5-inch A10X (2017)":
        "A1701, A1709, A1852",

    "iPad Pro 11-inch A12X (2018)":
        "A1980, A1979, A1934, A2013",

    "iPad Pro 11-inch A12Z (2020)":
        "A2228, A2068, A2230, A2231",

    "iPad Pro 11-inch M1 (2021)":
        "A2377, A2459, A2301, A2460",

    "iPad Pro 11-inch M2 (2022)":
        "A2759, A2761, A2435, A2762",

    "iPad Pro 11-inch M4 (2024)":
        "A2836, A2837, A3006",

    "iPad Pro 11-inch M4 (2024) Nano Glass":
        "A2836, A2837, A3006",

    "iPad Pro 11-inch M4 (2024) Standard Glass":
        "A2836, A2837, A3006",

    "iPad Pro 11-inch M5 (2025)":
        "A3357, A3358, A3359",

    "iPad Pro 12.9-inch A9X (2015)":
        "A1584, A1652",

    "iPad Pro 12.9-inch A10X (2017)":
        "A1670, A1671, A1821",

    "iPad Pro 12.9-inch A12X (2018)":
        "A1876, A2014, A1895, A1983",

    "iPad Pro 12.9-inch A12Z (2020)":
        "A2229, A2069, A2232, A2233",

    "iPad Pro 12.9-inch M1 (2021)":
        "A2378, A2461, A2379, A2462",

    "iPad Pro 12.9-inch M2 (2022)":
        "A2436, A2437, A2764, A2766",

    "iPad Pro 13-inch M4 (2024)":
        "A2925, A2926, A3007",

    "iPad Pro 13-inch M4 (2024) Nano Glass":
        "A2925, A2926, A3007",

    "iPad Pro 13-inch M4 (2024) Standard Glass":
        "A2925, A2926, A3007",

    "iPad Pro 13-inch M5 (2025)":
        "A3360, A3361, A3362",
}


# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

if "Standardized Model" not in df.columns:
    raise ValueError(
        "Column 'Standardized Model' was not found."
    )


# ------------------------------------------------------------
# Apply Model Number mappings
# ------------------------------------------------------------

df["Model Number"] = pd.NA

model_column = (
    df["Standardized Model"]
    .astype(str)
    .str.strip()
)

# iPhone
iphone_mask = (
    df["Device"]
    .astype(str)
    .str.strip()
    .eq("iPhone")
)

df.loc[iphone_mask, "Model Number"] = (
    model_column[iphone_mask]
    .map(iphone_model_numbers)
)

# iPad
ipad_mask = (
    df["Device"]
    .astype(str)
    .str.strip()
    .eq("iPad")
)

df.loc[ipad_mask, "Model Number"] = (
    model_column[ipad_mask]
    .map(ipad_model_numbers)
)


# ------------------------------------------------------------
# Preserve fictional test models as blank
# ------------------------------------------------------------

fictional_models = {
    "iPad Air 3",
    "iPad Air 4",
    "iPad Air 5",
    "iPhone 1",
    "iPhone 18 Pro Max",
}

df.loc[
    df["Standardized Model"].isin(fictional_models),
    "Model Number"
] = pd.NA

# Non-iPhone rows remain blank for now.
df.loc[~iphone_mask, "Model Number"] = pd.NA


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

iphone_models = (
    df.loc[iphone_mask, "Standardized Model"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
)

mapped = iphone_models[
    iphone_models.isin(iphone_model_numbers.keys())
]

unmapped = iphone_models[
    ~iphone_models.isin(iphone_model_numbers.keys())
]

print("=" * 60)
print("iPHONE MODEL NUMBER MAPPING")
print("=" * 60)

print(f"Unique iPhone models found : {len(iphone_models)}")
print(f"Mapped models              : {len(mapped)}")
print(f"Unmapped models            : {len(unmapped)}")

if len(unmapped) > 0:
    print("\nUNMAPPED iPHONE MODELS:")
    for model in unmapped:
        print(f"  - {model}")

print("\nFictional/test models intentionally excluded:")
for model in sorted(fictional_models):
    print(f"  - {model}")

# ------------------------------------------------------------
# Save new file — DO NOT overwrite original
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print(f"Saved: {OUTPUT_FILE}")
print("=" * 60)