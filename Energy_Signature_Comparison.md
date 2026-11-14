// ============================================
// ENERGY SIGNATURE COMPARISON
// ============================================
// Each signature has:
//   center: vector of length N (N = 10 in your case)
//   axis: vector of length N (forward direction, normalized)
//   points: list of linked points, each a vector of length N
//   weights: optional, how important each dimension / point is
// ============================================

FUNCTION CompareSignatures(sigA, sigB, tolerances):
    
    // STEP 1: Validate
    IF length(sigA.center) != length(sigB.center):
        RETURN { match: false, reason: "dimension mismatch" }
    
    N = length(sigA.center)

    // STEP 2: Normalize each signature to its own canonical frame
    // This is the key step. It makes comparison invariant to
    // translation and rotation.
    normA = NormalizeToCanonicalFrame(sigA)
    normB = NormalizeToCanonicalFrame(sigB)

    // STEP 3: Compare centers (should both be at origin after normalization)
    center_distance = EuclideanDistance(normA.center, normB.center)

    // STEP 4: Compare axes (forward / backward center line)
    // After normalization, both axes should point the same way.
    // Cosine similarity = 1 means same direction, -1 means opposite.
    axis_similarity = CosineSimilarity(normA.axis, normB.axis)

    // STEP 5: Compare the linkage structure
    // Compare the relationships between points, not just the points.
    // For each signature, build a linkage matrix:
    //   linkage[i][j] = distance and angle between point i and point j
    linkageA = BuildLinkageMatrix(normA.points)
    linkageB = BuildLinkageMatrix(normB.points)

    linkage_error = CompareLinkageMatrices(linkageA, linkageB)

    // STEP 6: Compare point-by-point if counts match
    // If the signatures have different numbers of linked points,
    // use a set-based distance instead.
    IF length(normA.points) == length(normB.points):
        point_error = 0
        FOR i IN 0 .. length(normA.points)-1:
            point_error += WeightedDistance(
                normA.points[i],
                normB.points[i],
                tolerances.weights
            )
        point_error = point_error / length(normA.points)
    ELSE:
        // Different counts: use Hausdorff-style distance
        point_error = HausdorffDistance(normA.points, normB.points)

    // STEP 7: Combine into a single score
    // Lower score = more similar
    total_score = (
        tolerances.w_center * center_distance +
        tolerances.w_axis * (1 - axis_similarity) +
        tolerances.w_linkage * linkage_error +
        tolerances.w_points * point_error
    )

    // STEP 8: Decide match / no-match
    is_match = total_score <= tolerances.threshold

    RETURN {
        match: is_match,
        score: total_score,
        details: {
            center_distance: center_distance,
            axis_similarity: axis_similarity,
            linkage_error: linkage_error,
            point_error: point_error
        }
    }
END FUNCTION

// ============================================
// HELPER: Normalize to canonical frame
// ============================================
FUNCTION NormalizeToCanonicalFrame(sig):
    
    // 1. Translate so spherical central point is at origin
    translated_points = []
    FOR p IN sig.points:
        translated_points.append(p - sig.center)
    
    translated_center = [0, 0, ..., 0] // length N

    // 2. Rotate so forward center line aligns to standard axis
    // Standard axis = [1, 0, 0, ..., 0]
    // Build a rotation matrix that maps sig.axis -> standard axis
    // In N dimensions this is a generalized rotation (Householder / QR)
    R = RotationMatrixFromTo(sig.axis, StandardAxis(N))
    
    rotated_points = []
    FOR p IN translated_points:
        rotated_points.append(R * p)
    
    rotated_axis = R * sig.axis  // should now be ~ [1, 0, 0, ...]

    RETURN {
        center: translated_center,
        axis: rotated_axis,
        points: rotated_points
    }
END FUNCTION

// ============================================
// HELPER: Build linkage matrix
// ============================================
FUNCTION BuildLinkageMatrix(points):
    // For each pair i,j, store distance and relative angle
    // This captures the *relationship* structure, which is
    // more stable than absolute positions.
    M = 2D array [len(points)][len(points)]
    FOR i IN 0 .. len(points)-1:
        FOR j IN 0 .. len(points)-1:
            IF i == j:
                M[i][j] = 0
            ELSE:
                M[i][j] = EuclideanDistance(points[i], points[j])
    RETURN M
END FUNCTION

// ============================================
// HELPER: Basic math
// ============================================
FUNCTION EuclideanDistance(a, b):
    sum = 0
    FOR k IN 0 .. length(a)-1:
        diff = a[k] - b[k]
        sum += diff * diff
    RETURN sqrt(sum)

FUNCTION CosineSimilarity(a, b):
    dot = DotProduct(a, b)
    magA = EuclideanDistance(a, ZeroVector(length(a)))
    magB = EuclideanDistance(b, ZeroVector(length(b)))
    IF magA == 0 OR magB == 0:
        RETURN 0
    RETURN dot / (magA * magB)

FUNCTION WeightedDistance(a, b, weights):
    sum = 0
    FOR k IN 0 .. length(a)-1:
        diff = a[k] - b[k]
        sum += weights[k] * diff * diff
    RETURN sqrt(sum)
