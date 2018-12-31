# convex-magic
Random sampling of convex polytopes

sample.py implements the billiard walk algorithm (1) to uniformly sample from
a convex polytope given by inequality constraints ``Ax <= b``.

According to the paper, this approach converges faster than standard hit-and-run
sampling (and can, for elongated polytopes or high dimensions, be much quicker 
than rejection sampling).

[1]: GRYAZINA, Elena; POLYAK, Boris. Random sampling: Billiard walk 
algorithm. European Journal of Operational Research, 2014, 238.2: 497-504.
