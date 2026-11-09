# Secondary support-bound diagnostic — before any reference join

Added after inference began but before model-reference scoring. This is a registered-late diagnostic, not a primary confirmatory endpoint or a new inference policy. No input, threshold, cohort or model changes.

Every saved raw mask belongs to one native1024view and is pasted only inside that view. Cross-view NMS retains masks without union. For nonempty valid reference Y and permitted view rectangles V, define

    U(Y) = max_V |Y intersect V| / |Y|.

For any mask M confined to one V, IoU(M,Y) <= |Y intersect V|/|Y| <= U(Y): its intersection is at most |Y intersect V| and its union contains Y. Equality for a particular V is attained by the privileged mask M=Y intersect V. This is an elementary support bound, not novelty, not an RGB mask predictor and not simultaneous instance recovery.

After all outputs are frozen, compute U for each published valid reference, count U<0.5/0.75 and relate it to actual common raw absence. Check that no actual raw IoU exceeds U beyond numerical tolerance. A bound below the threshold establishes impossibility **under the registered one-view mask materialization**, not RGB impossibility or an architectural bottleneck independent of tiling. A permissive bound does not establish learnability or correct identity.

The diagnostic tests whether altered physical view support excludes target geometry in this cohort. Do not automatically resize, enlarge views, merge masks or retune after results. Any future inference change would require a separate justified protocol; it cannot rewrite this frozen comparison.
