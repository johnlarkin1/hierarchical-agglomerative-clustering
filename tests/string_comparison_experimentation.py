import re


def preprocess_string(s):
    """Preprocess the string by removing spaces, converting to lowercase, and removing numbers."""
    s = s.replace(" ", "").lower()
    s = re.sub(r"\d+", "", s)
    return s


# Example usage
string1 = preprocess_string("Example Retailer 123")
string2 = preprocess_string("exampleRetailer")

import Levenshtein


def levenshtein_similarity(s1, s2):
    """Calculate normalized Levenshtein similarity between two strings."""
    distance = Levenshtein.distance(s1, s2)
    # Normalize by the length of the longer string
    max_len = max(len(s1), len(s2))
    similarity = 1 - (distance / max_len)
    return similarity


# Example usage
lev_similarity = levenshtein_similarity(string1, string2)


def jaro_winkler_similarity(s1, s2):
    """Calculate Jaro-Winkler similarity between two strings."""
    return Levenshtein.jaro_winkler(s1, s2)


# Example usage
jw_similarity = jaro_winkler_similarity(string1, string2)


def combined_similarity(lev_similarity, jw_similarity, lev_weight=0.5, jw_weight=0.5):
    """Combine the similarities using weighted average."""
    return lev_similarity * lev_weight + jw_similarity * jw_weight


# Example usage
combined_sim = combined_similarity(lev_similarity, jw_similarity, lev_weight=0.7, jw_weight=0.3)


def are_similar(combined_sim, threshold=0.8):
    """Determine if the combined similarity score indicates the strings are similar."""
    return combined_sim >= threshold


# Example usage
similar = are_similar(combined_sim, threshold=0.85)

string1 = "Example Retailer 123 Inc."
string2 = "ExmpleRetailr"

# Preprocess
string1_processed = preprocess_string(string1)
string2_processed = preprocess_string(string2)

# Calculate similarities
lev_similarity = levenshtein_similarity(string1_processed, string2_processed)
print("lev_similarity:", lev_similarity)
jw_similarity = jaro_winkler_similarity(string1_processed, string2_processed)
print("jw_similarity:", jw_similarity)
# Combine and decide
combined_sim = combined_similarity(lev_similarity, jw_similarity, lev_weight=0.7, jw_weight=0.3)
similar = are_similar(combined_sim, threshold=0.85)

print(f"Are the strings similar? {'Yes' if similar else 'No'}")
