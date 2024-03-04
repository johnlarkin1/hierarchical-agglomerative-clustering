combos = [
    ("simple", "levenshtein", "single", "scratch"),
    ("simple", "levenshtein", "single", "3rd party"),
    ("simple", "levenshtein", "complete", "scratch"),
    ("simple", "levenshtein", "complete", "3rd party"),
    ("simple", "levenshtein", "average", "scratch"),
    ("simple", "levenshtein", "average", "3rd party"),
    ("simple", "levenshtein", "ward", "scratch"),
    ("simple", "levenshtein", "ward", "3rd party"),
    ("simple", "levenshtein", "centroid", "scratch"),
    ("simple", "levenshtein", "centroid", "3rd party"),
    ("complex", "levenshtein", "single", "scratch"),
    ("complex", "levenshtein", "single", "3rd party"),
    ("complex", "levenshtein", "complete", "scratch"),
    ("complex", "levenshtein", "complete", "3rd party"),
    ("complex", "levenshtein", "average", "scratch"),
    ("complex", "levenshtein", "average", "3rd party"),
    ("complex", "levenshtein", "ward", "scratch"),
    ("complex", "levenshtein", "ward", "3rd party"),
    ("complex", "levenshtein", "centroid", "scratch"),
    ("complex", "levenshtein", "centroid", "3rd party"),
]

for combo in combos:
    file_type, dist_algo, linkage_type, should_use_third_party = combo
    dist_algo = "leven"
    impl_type = should_use_third_party
    core_name = f"dendrogram_{impl_type}_{file_type}_{linkage_type}_{dist_algo}"
    print(f"{core_name}.png")

input_to_image_map = {
    # EASY, 3rd Party
    (
        "DATA_FILE_PATH_EASY",
        True,
        False,
        "SINGLE",
    ): "/images/hac/dendrogram_3rd_party_simple_single_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        True,
        False,
        "COMPLETE",
    ): "/images/hac/dendrogram_3rd_party_simple_complete_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        True,
        False,
        "AVERAGE",
    ): "/images/hac/dendrogram_3rd_party_simple_average_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        True,
        False,
        "CENTROID",
    ): "/images/hac/dendrogram_3rd_party_simple_centroid_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        True,
        False,
        "WARD",
    ): "/images/hac/dendrogram_3rd_party_simple_ward_leven.png",
    # CPX, 3rd Party
    (
        "DATA_FILE_PATH_CPX",
        True,
        False,
        "SINGLE",
    ): "/images/hac/dendrogram_3rd_party_complex_single_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        True,
        False,
        "COMPLETE",
    ): "/images/hac/dendrogram_3rd_party_complex_complete_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        True,
        False,
        "AVERAGE",
    ): "/images/hac/dendrogram_3rd_party_complex_average_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        True,
        False,
        "CENTROID",
    ): "/images/hac/dendrogram_3rd_party_complex_centroid_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        True,
        False,
        "WARD",
    ): "/images/hac/dendrogram_3rd_party_complex_ward_leven.png",
    # EASY, From scratch
    (
        "DATA_FILE_PATH_EASY",
        False,
        True,
        "SINGLE",
    ): "/images/hac/dendrogram_scratch_simple_single_leven.p",
    (
        "DATA_FILE_PATH_EASY",
        False,
        True,
        "COMPLETE",
    ): "/images/hac/dendrogram_scratch_simple_complete_leven.p",
    (
        "DATA_FILE_PATH_EASY",
        False,
        True,
        "AVERAGE",
    ): "/images/hac/dendrogram_scratch_simple_average_leven.p",
    (
        "DATA_FILE_PATH_EASY",
        False,
        False,
        "SINGLE",
    ): "/images/hac/dendrogram_scratch_simple_single_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        False,
        False,
        "COMPLETE",
    ): "/images/hac/dendrogram_scratch_simple_complete_leven.png",
    (
        "DATA_FILE_PATH_EASY",
        False,
        False,
        "AVERAGE",
    ): "/images/hac/dendrogram_scratch_simple_average_leven.png",
    # CPX, From scratch
    (
        "DATA_FILE_PATH_CPX",
        False,
        True,
        "SINGLE",
    ): "/images/hac/dendrogram_scratch_complex_single_leven.p",
    (
        "DATA_FILE_PATH_CPX",
        False,
        True,
        "COMPLETE",
    ): "/images/hac/dendrogram_scratch_complex_complete_leven.p",
    (
        "DATA_FILE_PATH_CPX",
        False,
        True,
        "AVERAGE",
    ): "/images/hac/dendrogram_scratch_complex_average_leven.p",
    (
        "DATA_FILE_PATH_CPX",
        False,
        False,
        "SINGLE",
    ): "/images/hac/dendrogram_scratch_complex_single_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        False,
        False,
        "COMPLETE",
    ): "/images/hac/dendrogram_scratch_complex_complete_leven.png",
    (
        "DATA_FILE_PATH_CPX",
        False,
        False,
        "AVERAGE",
    ): "/images/hac/dendrogram_scratch_complex_average_leven.png",
}
image_to_input_map = {value: key for key, value in input_to_image_map.items()}

execution_times = [
    2.8021,
    5.2054,
    2.7315,
    1.7993,
    2.2091,
    2.0232,
    2.1442,
    1.9224,
    3.7413,
    3.3757,
    1.4791,
    1.2263,
    1.5507,
    44.7954,
    27.1838,
    72.2544,
    12276.2146,
    556.8679,
    613.8062,
    87855.6781,
    7232.4973,
    12005.9713,
]

target_output_order = [
    "/images/hac/dendrogram_scratch_simple_single_leven.png",
    "/images/hac/dendrogram_3rd_party_simple_single_leven.png",
    "/images/hac/dendrogram_scratch_simple_complete_leven.png",
    "/images/hac/dendrogram_3rd_party_simple_complete_leven.png",
    "/images/hac/dendrogram_scratch_simple_average_leven.png",
    "/images/hac/dendrogram_3rd_party_simple_average_leven.png",
    "/images/hac/dendrogram_3rd_party_simple_ward_leven.png",
    "/images/hac/dendrogram_3rd_party_simple_centroid_leven.png",
    "/images/hac/dendrogram_scratch_complex_single_leven.png",
    "/images/hac/dendrogram_3rd_party_complex_single_leven.png",
    "/images/hac/dendrogram_scratch_complex_complete_leven.png",
    "/images/hac/dendrogram_3rd_party_complex_complete_leven.png",
    "/images/hac/dendrogram_scratch_complex_average_leven.png",
    "/images/hac/dendrogram_3rd_party_complex_average_leven.png",
    "/images/hac/dendrogram_3rd_party_complex_ward_leven.png",
    "/images/hac/dendrogram_3rd_party_complex_centroid_leven.png",
]

image_to_execution_time = {
    image_name: execution_time
    for image_name, execution_time in zip(image_to_input_map, execution_times)
}


for output in target_output_order:
    print("output", output)
    print("execution_time", image_to_execution_time[output])
